from neuralforecast.auto import (AutoNHITS,
                                 AutoMLP,
                                 AutoLSTM,
                                 AutoTFT,
                                 AutoPatchTST,
                                 AutoKAN)

from neuralforecast.losses.pytorch import DistributionLoss, MAE


def get_auto_nf_models(horizon, loss: str, rs_n_samples: int):
    NEED_CPU = ['AutoLSTM', 'AutoKAN', 'AutoMLP',
                'AutoNHITS', 'AutoTFT', 'AutoPatchTST']

    model_cls = {
        'AutoKAN': AutoKAN,
        'AutoMLP': AutoMLP,
        'AutoNHITS': AutoNHITS,
        # 'AutoTFT': AutoTFT,
        'AutoPatchTST': AutoPatchTST,
        'AutoLSTM': AutoLSTM,
    }

    if loss == 'poisson':
        loss_inst = DistributionLoss(distribution='Poisson', level=[80, 90], return_params=False)
    elif loss == 'tweedie':
        loss_inst = DistributionLoss(distribution='Tweedie', level=[80, 90], rho=1.5, return_params=False)
    else:
        loss_inst = MAE()

    models = []
    for mod_name, mod in model_cls.items():
        if mod_name in NEED_CPU:
            mod.default_config['accelerator'] = 'cpu'
        else:
            mod.default_config['accelerator'] = 'mps'

        model_instance = mod(
            loss=loss_inst,
            h=horizon,
            num_samples=rs_n_samples,
            alias=mod_name,
        )

        models.append(model_instance)

    return models
