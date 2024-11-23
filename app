import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from prophet import Prophet
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Analisis Socioeconomico Mundial", page_icon="🌍")

st.title("Analisis de Indicadores Globales")
st.markdown("<h3 style='color: red;'>Visualiza y examina indicadores clave para diferentes paises.</h3>", unsafe_allow_html=True)

@st.cache_data
def cargar_datos(codigo_indicador, años_inicio, años_final):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{codigo_indicador}"
    parametros = {"format": "json", "date": f"{años_inicio}:{años_final}", "per_page": "1000"}
    respuesta = requests.get(url, params=parametros)
    if respuesta.status_code == 200:
        datos = respuesta.json()[1]
        if datos:
            return pd.DataFrame([
                {"Pais": fila["country"]["value"], "Codigo": fila["countryiso3code"], "Año": fila["date"], "Valor": fila["value"]}
                for fila in datos if fila["value"] is not None
            ])
    return pd.DataFrame()

indicadores = {
    "Producto Interno Bruto": "NY.GDP.PCAP.CD",
    "Esperanza de Vida": "SP.DYN.LE00.IN",
    "Poblacion Total": "SP.POP.TOTL",
    "Alfabetizacion": "SE.ADT.LITR.ZS"
}

rango_años = (1960, 2020)
dataframes = {key: cargar_datos(value, *rango_años) for key, value in indicadores.items()}

if any(df.empty for df in dataframes.values()):
    st.markdown("<h4 style='color: red;'>No se encontraron datos disponibles. Por favor, intenta con otra opcion.</h4>", unsafe_allow_html=True)
    st.stop()

st.sidebar.header("Configuracion de Visualizacion")
pais = st.sidebar.selectbox("Selecciona un pais", dataframes["Producto Interno Bruto"]["Pais"].unique())
indicador = st.sidebar.selectbox("Selecciona un indicador", list(indicadores.keys()))
prediccion = st.sidebar.checkbox("Incluir prediccion")

años_inicio = st.sidebar.text_input("Escribe el año de inicio", value="2000")
años_final = st.sidebar.text_input("Escribe el año de final", value="2020")

if not (años_inicio.isdigit() and años_final.isdigit()):
    st.markdown("<h4 style='color: red;'>Por favor ingresa años validos.</h4>", unsafe_allow_html=True)
    st.stop()

años_inicio, años_final = int(años_inicio), int(años_final)
datos = cargar_datos(indicadores[indicador], años_inicio, años_final)
datos_filtrados = datos[datos["Pais"] == pais]

st.subheader(f"{indicador} para {pais}")
grafico = px.line(datos_filtrados, x="Año", y="Valor", labels={"Valor": f"{indicador}"}, title=f"Evolucion de {indicador} en {pais}")
grafico.update_traces(mode="lines+markers")
st.plotly_chart(grafico)

if prediccion:
    datos_prediccion = datos_filtrados.rename(columns={"Año": "ds", "Valor": "y"})
    if len(datos_prediccion) > 5:
        modelo = Prophet()
        modelo.fit(datos_prediccion)
        futuro = modelo.make_future_dataframe(periods=5, freq="Y")
        prediccion = modelo.predict(futuro)
        grafico_prediccion = px.line(prediccion, x="ds", y="yhat", title=f"Prediccion para {pais} en {indicador}")
        grafico_prediccion.add_scatter(x=datos_prediccion["ds"], y=datos_prediccion["y"], mode="markers", name="Datos Actuales")
        st.plotly_chart(grafico_prediccion)
    else:
        st.markdown("<h4 style='color: red;'>No hay datos suficientes para realizar una prediccion.</h4>", unsafe_allow_html=True)

st.subheader("Comparacion Regional")
año_regional = st.sidebar.text_input("Escribe un año para comparar regiones", value="2020")

if not año_regional.isdigit():
    st.markdown("<h4 style='color: red;'>Por favor ingresa un año valido para el mapa.</h4>", unsafe_allow_html=True)
    st.stop()

año_regional = int(año_regional)
datos_anuales = datos[datos["Año"] == str(año_regional)]

mapa_nuevo = folium.Map(location=[0, 0], zoom_start=3)
for _, fila in datos_anuales.iterrows():
    folium.Marker(location=[0, 0], popup=f"{fila['Pais']}: {fila['Valor']}").add_to(mapa_nuevo)

st.subheader("Mapa Global Actualizado")
st_folium(mapa_nuevo, width=700)
