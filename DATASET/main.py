import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from datetime import date,timedelta

import plotly.express as px

from yellowbrick.cluster import KElbowVisualizer,SilhouetteVisualizer,InterclusterDistance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df_orders = pd.read_csv('Brazilian E-Commerce Public Dataset/olist_orders_dataset.csv')
df_payments = pd.read_csv('Brazilian E-Commerce Public Dataset/olist_order_payments_dataset.csv')

#visualiza as colunas e linhas 
df_orders.shape

#visualiza as colunas, somente
df_orders.columns

#visualiza o tipo dos dados que formam o dataset
df_orders.dtypes

#pegando uma amosta do dataset para conferir..

df_orders.head(5)

#convert ['order_purchase_timestamp','order_approved_at', 'order_delivered_carrier_date','order_delivered_customer_date', 'order_estimated_delivery_date'] to datetime
date_columns = ['order_purchase_timestamp','order_approved_at', 'order_delivered_carrier_date','order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_columns:
    df_orders[col] = pd.to_datetime(df_orders[col])
df_orders.dtypes

# verificando números nulos
df_orders.isna().sum()

# verificando o status de cada pedido e a quantidade por cada status,
df_orders['order_status'].value_counts()

# Filtrando somente pelos pedidos ENTREGUES, somente.
df_delivered = df_orders.query('order_status == "delivered"')

# e vefica se tem "NaN" nesse dataset após o filtro 
df_delivered.isna().sum()

#list columns
df_payments.columns

# verifica se tem pedidos duplicados
df_payments.duplicated().value_counts()

# verificando a distribuição pelo valor de pagamento
sns.displot(df_payments['payment_value'],bins=20)


# JUNTANDO PAGAMENTOS E PEDIDOS
# convertendo o 'order_id' para index nos dois datasets
df_payments = df_payments.set_index('order_id')
df_orders = df_orders.set_index('order_id')
order_payment = df_orders.join(df_payments)


#FINALMENTE, gerando o RFM DATAFRAME

last_date = order_payment['order_delivered_carrier_date'].max() + timedelta(days=1)

rfm = order_payment.groupby('customer_id').agg({
    'order_delivered_carrier_date': lambda x: (last_date - x.max()).days,
    'order_id': lambda x: len(x),
    'payment_value': 'sum'
})

rfm.dropna(inplace=True)
std = StandardScaler()
x_std = std.fit_transform(rfm)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(x_std)        
visualizer.show()


model_k = KMeans(n_clusters=4)
kmeans = model_k.fit(x_std)
rfm['cluster'] = kmeans.labels_

rfm.columns = ['Recency','Frequency','MonetaryValue','cluster']

rfm.head()
