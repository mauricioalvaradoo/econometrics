import pandas as pd
import requests



def get_data(series, fechaini, fechafin):

    """ Importar multiples series de la API del BCRP
    
    Parámetros
    ----------
    series: dict
        Diccionario de los códigos y nombres de las series
    fechaini: datetime
        Fecha de inicio de la serie:
        - Diario: yyyy-mm-dd
        - Mensual: yyyy-mm
        - Trimestral: yyyy-q
        - Anual: yyyy
    fechafin: datetime
        Fecha de fin de la serie:
        - Diario: yyyy-mm-dd
        - Mensual: yyyy-mm
        - Trimestral: yyyy-q 
        - Anual: yyyy
        
    Retorno
    ----------
    df: pd.DataFrame
        Series consultadas
    
    Documentación
    ----------
    https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/api
    
    @author: Mauricio Alvarado
    
    """

    keys = list(series.keys())
    
    
    df = pd.DataFrame()
    base = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
        

    for i in keys:
        url = f"{base}/{i}/json/{fechaini}/{fechafin}/ing"

        r = requests.get(url)
        
        if r.status_code == 200:
            pass
        else:
            print("Vinculacion inválida!")
            break
        
        response = r.json().get("periods")
        
        list_values = []
        list_time = []
                
        for j in response:
            list_values.append(float(j["values"][0]))    
            list_time.append(j["name"])

        # Merge
        dic = pd.DataFrame({"time": list_time, f"{i}": list_values})                      
        df = pd.concat([df, dic]) if df.empty is True else pd.merge(df, dic, how="outer")
        
    df.set_index("time", inplace=True)
    df.rename(series, axis=1, inplace=True)


    # Modificaciones adicionales
    try:
        df.index = pd.period_range(fechaini, fechafin, freq="Q") # Trimestral
    except:
        try:
            df.index = df.index.str.replace('Set', 'Sep')
            df.index = pd.to_datetime(df.index, format="%b.%Y") # Mensuales
        except:
            try:
                df.index = pd.to_datetime(df.index, format="%d.%b.%y") # Diarias
            except:
                pass

    return df