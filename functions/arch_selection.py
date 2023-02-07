from arch import arch_model


def model_selection(serie):
    arch_1_fit = arch_model(serie, p=1, q=0).fit(disp=0)
    arch_2_fit = arch_model(serie, p=2, q=0).fit(disp=0)
    arch_3_fit = arch_model(serie, p=3, q=0).fit(disp=0)
    arch_4_fit = arch_model(serie, p=4, q=0).fit(disp=0)
    arch_5_fit = arch_model(serie, p=5, q=0).fit(disp=0)
    garch_11_fit = arch_model(serie, p=1, q=1).fit(disp=0)
    garch_12_fit = arch_model(serie, p=1, q=2).fit(disp=0)
    garch_13_fit = arch_model(serie, p=1, q=2).fit(disp=0)
    garch_21_fit = arch_model(serie, p=2, q=1).fit(disp=0)
    garch_22_fit = arch_model(serie, p=2, q=2).fit(disp=0)
    garch_23_fit = arch_model(serie, p=2, q=2).fit(disp=0)
    garch_31_fit = arch_model(serie, p=2, q=2).fit(disp=0)
    garch_32_fit = arch_model(serie, p=2, q=2).fit(disp=0)
    garch_33_fit = arch_model(serie, p=2, q=2).fit(disp=0)


    aic_models = {
        "ARCH (1)": arch_1_fit.aic,
        "ARCH (2)": arch_2_fit.aic,
        "ARCH (3)": arch_3_fit.aic,
        "ARCH (4)": arch_4_fit.aic,
        "ARCH (5)": arch_5_fit.aic,
        "GARCH (1, 1)": garch_11_fit.aic,
        "GARCH (1, 2)": garch_12_fit.aic,
        "GARCH (1, 3)": garch_13_fit.aic,
        "GARCH (2, 1)": garch_21_fit.aic,
        "GARCH (2, 2)": garch_22_fit.aic,
        "GARCH (2, 3)": garch_23_fit.aic,
        "GARCH (3, 1)": garch_31_fit.aic,
        "GARCH (3, 2)": garch_32_fit.aic,
        "GARCH (3, 3)": garch_33_fit.aic,
        }

    bic_models = {
        "ARCH (1)": arch_1_fit.bic,
        "ARCH (2)": arch_2_fit.bic,
        "ARCH (3)": arch_3_fit.bic,
        "ARCH (4)": arch_4_fit.bic,
        "ARCH (5)": arch_5_fit.bic,
        "GARCH (1, 1)": garch_11_fit.bic,
        "GARCH (1, 2)": garch_12_fit.bic,
        "GARCH (1, 3)": garch_13_fit.bic,
        "GARCH (2, 1)": garch_21_fit.bic,
        "GARCH (2, 2)": garch_22_fit.bic,
        "GARCH (2, 3)": garch_23_fit.bic,
        "GARCH (3, 1)": garch_31_fit.bic,
        "GARCH (3, 2)": garch_32_fit.bic,
        "GARCH (3, 3)": garch_33_fit.bic,
    }

    aic_selected = None
    bic_selected = None

    for i in aic_models.keys():
        if aic_selected == None:
            aic_selected = aic_models[i]
            aic_model = i
        else:
            x = aic_models[i]
            if aic_selected > x:
                aic_selected = x
                aic_model = i
            
    for i in bic_models.keys():
        if bic_selected == None:
            bic_selected = bic_models[i]
            bic_model = i
        else:
            x = bic_models[i]
            if bic_selected > x:
                bic_selected = x
                bic_model = i
        
    return print(f"El modelo seleccionado es {aic_model} segun AIC y {bic_model} segun el BIC.")