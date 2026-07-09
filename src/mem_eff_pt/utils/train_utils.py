import torch
import torch.nn as nn
from vonsoap.optimizers import SOAP, IVON, EVON, AdamWBF16

def build_model(model, args):
    return model

def build_optimizer(model, trainable_params, args):   
    main_params = []
    oned_params = []
    secondary_params = []
    main_modules_list = ["attn", "mlp","attention"]
    #main_modules_list = ["attn", "mlp","attention", "embed_tokens"]
    main_params_old = []

    print(f"MAIN MODULES = {main_modules_list} !!!")
    id_to_name_main_params = {}
    id_to_name_secondary_params = {}
    id_to_name_oned_params = {}

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
        #if not (isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            continue
        if not any(target_key in module_name for target_key in main_modules_list):
            continue

        main_params.append(module.weight)
        id_to_name_main_params[id(module.weight)] = module_name

    merge_info = {}
    flip_info = {}
    for param_name, p in model.named_parameters():
        if args.optimizer.lower().find('_kv_merged')>0:
            if param_name.find('attn.k_proj')>=0 or param_name.find('attn.v_proj')>=0:
                name = param_name.split('.')
                kname = '%s.kv.%s'%('.'.join(name[:-2]), '.'.join(name[-1:]))
                print('merging:', name, kname, p.shape)
                li = merge_info.setdefault(kname, [])
                li.append(p)
                continue

        if args.optimizer.lower().find('_mlp_merged')>0:
            #if param_name.find('gate_proj')>=0 or param_name.find('up_proj')>=0 or (args.optimizer.lower().find('_noflip')<0 and param_name.find('down_proj')>=0):
            if param_name.find('gate_proj')>=0 or param_name.find('up_proj')>=0:
                name = param_name.split('.')
                kname = '%s.w12.%s'%('.'.join(name[:-2]), '.'.join(name[-1:]))
                print('merging:', name, kname, p.shape)
                if param_name.find('down_proj')>=0:
                    flip_info.setdefault(kname, p)
                else:
                    li = merge_info.setdefault(kname, [])
                    li.append(p)
                continue

        if id(p) in id_to_name_main_params:
            main_params_old.append(p)
            continue

        if p.ndim == 1 or param_name.find('lm_head')>=0 or param_name.find('embed')>=0:
        #if p.ndim == 1:
            oned_params.append(p)
            id_to_name_oned_params[id(p)] = param_name
            #print('adamw',  param_name, p.shape)
        else:
            secondary_params.append(p)
            main_params_old.append(p)
            id_to_name_secondary_params[id(p)] = param_name

    main_params = []
    main_params.append( {'params':main_params_old,
                        'weight_decay':args.weight_decay,
                         } )
    count_merged = 0
    for key, params in merge_info.items():
        count_merged += len(params)
        if key in flip_info:
            flip_param = flip_info[key]
            count_merged += 1
            params.append(flip_param)
            main_params.append( {'params':params, 'merged':True, 'weight_decay':args.weight_decay, 'flip_end':True } )
        else:
            main_params.append( {'params':params, 'merged':True, 'weight_decay':args.weight_decay} )

    assert len(trainable_params) == count_merged + len(main_params_old) + len(oned_params)


    elif args.optimizer.lower().find('soap')>=0:
        opt_name = args.optimizer.lower()
        cast_dtype = torch.bfloat16
        if opt_name.find('_fp32')>=0:
            cast_dtype = torch.float32

        improve_orth = False
        if opt_name.find('_improve')>=0:
            improve_orth = True
 
        optimizer1 = SOAP(
                main_params,
                lr=args.lr,
                betas=(args.momentum, 1.0-args.lr_cov),
                correct_bias=False,
                eps = args.damping,
                precondition_1d = False,
                normalize_grads = False,
                weight_decay=args.weight_decay,
                precondition_frequency=args.freq,
                improve_orth = improve_orth,
                cast_dtype = cast_dtype,
        )

        optimizer2 = AdamWBF16(
            oned_params,
            lr=args.adam_lr,
            weight_decay=args.adam_weight_decay,
            betas=(args.adam_beta_1, args.adam_beta_2),
            eps=args.adam_damping,
            cast_dtype=torch.bfloat16,
            is_normalize = False,
        )

        print(optimizer1, optimizer2)
        return [optimizer1, optimizer2]

    elif args.optimizer.lower().find('ivon')>=0:
        optimizer1 = IVON(
            main_params_old,
            lr=args.lr,
            beta1=args.momentum,
            beta2=1.0 - args.lr_cov,
            ess=args.ess,
            hess_init=args.ivon_hess_init,
            clip_radius=args.ivon_clip_radius,
            weight_decay=args.weight_decay,
            sync=args.von_sync,
        )

        optimizer2 = AdamWBF16(
            oned_params,
            lr=args.adam_lr,
            weight_decay=args.adam_weight_decay,
            betas=(args.adam_beta_1, args.adam_beta_2),
            eps=args.adam_damping,
            cast_dtype=torch.bfloat16,
            is_normalize = False,
        )

        print(optimizer1, optimizer2)
        return [optimizer1, optimizer2]

    elif args.optimizer.lower().find('evon')>=0:
        optimizer1 = EVON(
            main_params,
            lr=args.lr,
            betas=(args.momentum, 1.0 - args.lr_cov),
            hess_init=args.ivon_hess_init,
            precondition_frequency=args.freq,
            ess=args.ess,
            correct_bias=True,
            eps=args.damping,
            max_precond_dim=args.max_precond_dim,
            weight_decay=args.weight_decay,
            precondition_1d=False,
            cast_dtype=args.cast_dtype,
            shampoo_beta=-1 if args.shampoo_beta is None else args.shampoo_beta,
            sync=args.von_sync,
            whiten_grad=args.whiten_evon_grad,
            price_clip_ratio=args.price_clip_ratio,
            phasing=args.evon_phased_grads
        )

        optimizer2 = AdamWBF16(
            oned_params,
            lr=args.adam_lr,
            weight_decay=args.adam_weight_decay,
            betas=(args.adam_beta_1, args.adam_beta_2),
            eps=args.adam_damping,
            cast_dtype=torch.bfloat16,
            is_normalize = False,
        )

        print(optimizer1, optimizer2)
        return [optimizer1, optimizer2]

    elif args.optimizer.lower() == "adamw":
        optimizer = AdamWBF16(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.momentum, 1.0 - args.lr_cov),
            eps=args.damping,
            cast_dtype=torch.bfloat16,
            is_normalize = False,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return [optimizer]
