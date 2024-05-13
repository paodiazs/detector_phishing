import pandas as pd
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

import os
import seaborn  as sns


df = pd.read_csv('/Users/paoladiaz/Desktop/url_detection/malicious_phish.csv') #leemos el dataset
#print("Dataset: ", df.shape) #devuelve una tupla que es el numero de filas y columnas
df.head() #muestra las primeras 5 lineas del dataset

df.type.value_counts() #devuelve los tipos y cuántos son de cada uno



import re #modulo de expresiones regulares de python
def having_ip_address(url):
     #busca un patron en la URL para buscar direcciones ipv4 o ipv6
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match: #si encuentra, regresa 1
        return 1
    else: #si no encuentra, regresa 0
        return 0
#Crea una nueva columna llamada use_of_ip
df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))


#Tener diferentes tipos de keywords o algo que no es normalmente visto en urls
from urllib.parse import urlparse # se utiliza para analizar (parsear) una URL dada y dividirla en sus componentes como esquema, red, ubicación, ruta, parámetros, consulta, fragmento, etc.

def abnormal_url(url):
  hostname = urlparse(url).hostname #de urlparse tomamos hostname
  hostname = str(hostname) #Convierte hostname en una cadena (string) si es necesario, para asegurarnos de que sea un tipo de datos compatible con re.search.
  match = re.search(hostname, url) #busca re.search apra buscar que hostname esté
  if match: #si si está
    return 1 #regresa1
  else:
    return 0 #sino, regresa 0
#utiliza apply para aplicar la función abnormal_url a cada elemento (i) en la columna 'url' del DataFrame df.
df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))

## CONTAR LOS PUNTOS EN LA URL
#La función devuelve el recuento de puntos dentro de la url


def count_dot(url):#funcion y le pasamos url
  count_dot = url.count('.') #hacemos un count de los puntos
  return count_dot #devolvemos el recuento

df['count-dot'] = df['url'].apply(lambda i: count_dot(i)) #añadimos la columna al dataset
df.head() #mostramos las primeras 5 lineas del dataset

"""## RECUENTO DE PALABRAS CLAVE `www`, `@`"""

def count_www(url):
  url.count('www')
  return url.count('www')
df['count-www'] = df['url'].apply(lambda i: count_www(i))

def count_atrate(url):
    return url.count('@')


df['count@'] = df['url'].apply(lambda i: count_atrate(i))

"""## NÚMERO DE DIRECTORIOS, dados por `/`"""

def no_of_dir(url):
  urldir = urlparse(url).path
  return urldir.count('/')
df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))

"""## NÚMERO DE INTEGRACIONES `//`"""

def no_of_embed(url):
  urldir = urlparse(url).path
  return urldir.count('//')
df['count_embed_domain'] = df['url'].apply(lambda i: no_of_embed(i))

"""## SHORTENING SERVICE
Hay servicios donde acortan la url para hacerla más pequeña
La función lo sidentificará y dirá si son urls acortadas o no
"""

def shortening_service(url):
    match = re.search('bit\\.ly|goo\\.gl|shorte\\.st|go2l\\.ink|x\\.co|ow\\.ly|t\\.co|tinyurl|tr\\.im|is\\.gd|cli\\.gs|'
                      'yfrog\\.com|migre\\.me|ff\\.im|tiny\\.cc|url4\\.eu|twit\\.ac|su\\.pr|twurl\\.nl|snipurl\\.com|'
                      'short\\.to|BudURL\\.com|ping\\.fm|post\\.ly|Just\\.as|bkite\\.com|snipr\\.com|fic\\.kr|loopt\\.us|'
                      'doiop\\.com|short\\.ie|kl\\.am|wp\\.me|rubyurl\\.com|om\\.ly|to\\.ly|bit\\.do|t\\.co|lnkd\\.in|'
                      'db\\.tt|qr\\.ae|adf\\.ly|goo\\.gl|bitly\\.com|cur\\.lv|tinyurl\\.com|ow\\.ly|bit\\.ly|ity\\.im|'
                      'q\\.gs|is\\.gd|po\\.st|bc\\.vc|twitthis\\.com|u\\.to|j\\.mp|buzurl\\.com|cutt\\.us|u\\.bb|yourls\\.org|'
                      'x\\.co|prettylinkpro\\.com|scrnch\\.me|filoops\\.info|vzturl\\.com|qr\\.net|1url\\.com|tweez\\.me|v\\.gd|'
                      'tr\\.im|link\\.zip\\.net',
                      url)
    if match:
        return 1
    else:
        return 0



df['short_url'] = df['url'].apply(lambda i: shortening_service(i))

"""## CONTADOR DE `HTTPS` Y `HTTP`"""

def count_https(url):
  return url.count('https')
df['count-https'] = df['url'].apply(lambda i: count_https(i))

def count_http(url):
  return url.count('http')
df['count-http'] = df['url'].apply(lambda i: count_http(i))

"""## CONTADOR DE `%`,` ?`,` - `, `=`
- Los % SON ESPACIOS
- ? normalmente un id

"""

def count_per(url):
  return url.count('%')
df['count%'] = df['url'].apply(lambda i: count_per(i))

def count_ques(url):
  return url.count('?')
df['count?'] = df['url'].apply(lambda i: count_ques(i))

def count_hyphen(url):
  return url.count('-')
df['count-'] = df['url'].apply(lambda i: count_hyphen(i))

def count_equal(url):
  return url.count('=')
df['count='] = df['url'].apply(lambda i: count_equal(i))

"""## LONGITUD DE LA URL"""

def url_length(url):
  return len(str(url))

df['url_length'] = df['url'].apply(lambda i: url_length(i))

"""LONGITUD DEL HOSTNAME"""

def hostname_length(url):
  return len(urlparse(url).netloc) #NETCLOC: Representa el nombre de host (hostname) y el número de puerto, si está presente, de la URL.

df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))

"""## PALABRAS SOSPECHOSAS"""

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',url)
    if match:
        return 1
    else:
        return 0
df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))

"""## CONTADOR DE DIGITOS Y DE LETRAS"""

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
df['count-digits'] = df['url'].apply(lambda i: digit_count(i))

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
df['count-letters'] = df['url'].apply(lambda i: letter_count(i))

"""## TOP LEVEL DOMAIN, Y PRIMER DIRECTORIO"""


from urllib.parse import urlparse
from tld import get_tld
import os.path

#Longitud del primer directorio
def fd_length(url):
    urlpath=urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
      return 0
df['fd_length'] = df['url'].apply(lambda i: fd_length(i))


#Longitud del top level domain
#crear una nueva columna llamada tld, que extrae el dominio superior y devuelve su longitud
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))

df = df.drop("tld", axis=1) #eliminamos la columna tld porque solo queremos valores numéricso

"""
Codificación del objetivo
nuestro objetivo es la columna tipo,vamos a convertir los nombres(secure,phishing,defacement,malware) en valores númericos
"""

from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df['type_code'] = lb_make.fit_transform(df['type'])
df['type_code'].value_counts()

"""# Creación de características y objetivo"""

#Predictor Variables
# filtering out google_index as it has only 1 value
X = df[['use_of_ip','abnormal_url', 'count-dot', 'count-www', 'count@',
       'count_dir', 'count_embed_domain', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable
y = df['type_code']

X.head()

"""# Training test SPLIT"""

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)

"""- stratify garantiza que las proporciones de clases en y se mantengan en las divisiones de entrenamiento y prueba.
- test_size Especifica el tamaño del conjunto de prueba. Aquí, el 20% de los datos se utilizarán como conjunto de prueba y el 80% restante como conjunto de entrenamiento.
- shuffle Indica si se deben mezclar los datos antes de dividirlos. En este caso, se mezclan los datos antes de la división.
- random_state Es una semilla (seed) utilizada por el generador de números aleatorios para garantizar que los resultados sean reproducibles. Estableciendo este valor, aseguras que cada vez que ejecutes esta división, obtendrás la misma división de datos.

# Construcción del modelo



"""# 2. XGBoost Classifier"""

xgb_c = xgb.XGBClassifier(n_estimators=100)
xgb_c.fit(X_train, y_train)
y_pred_x = xgb_c.predict(X_test)
#print(classification_report(y_test, y_pred_x, target_names=['benign', 'defacement', 'phishing', 'malware']))



"""# Predicción"""

def main(url):
  status = []

  status.append(having_ip_address(url))
  status.append(abnormal_url(url))
  status.append(count_dot(url))
  status.append(count_www(url))
  status.append(count_atrate(url))
  status.append(no_of_dir(url))
  status.append(no_of_embed(url))

  status.append(shortening_service(url))
  status.append(count_https(url))
  status.append(count_http(url))

  status.append(count_per(url))
  status.append(count_ques(url))
  status.append(count_hyphen(url))
  status.append(count_equal(url))

  status.append(url_length(url))
  status.append(hostname_length(url))
  status.append(suspicious_words(url))
  status.append(digit_count(url))
  status.append(letter_count(url))

  status.append(fd_length(url))
  status.append(tld_length(get_tld(url,fail_silently=True)))

  return status


def get_prediction_from_url(test_url):
  features_test = main(test_url)
  features_test = np.array(features_test).reshape(1, -1)

  pred = xgb_c.predict(features_test)
  if int(pred[0]) == 0:
    res = "SAFE"
    return res
  elif int(pred[0]) == 1.0:
    res = "DEFACEMENT"
    return res
  elif int(pred[0]) == 2.0:
    res = "PHISHING"
    return res
  elif int(pred[0]) == 3.0:
    res = "PHISHING"
    return res

urls = []
for url in urls:
  print(get_prediction_from_url(url))