import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# import logging
# logging.getLogger().setLevel(logging.ERROR)

st.set_page_config(layout='wide', page_title="Klasyfikacja obiektów w kosmosie")


@st.cache_data  # funkcja będzie zapamiętywać wynik funkcji w cache
def load_data():
    data = pd.read_csv("star_classification.csv")
    data = data.dropna(subset=['redshift',  'u', 'g', 'r', 'i', 'z', 'class'])
    return data


df = load_data()
df = df.drop(index=79543)
df_cleaned = df[['redshift', 'u', 'g', 'r', 'i', 'z', 'class']]
st.sidebar.title("Nawigacja")
page = st.sidebar.radio("Przejdź do:", ["Wprowadzenie", "Eksploracja danych", "Model"])

# -------------------
# 1. WPROWADZENIE
# -------------------

if page == "Wprowadzenie":
    st.title("Wprowadzenie")
    st.markdown("""
    Ten dashboard pozwala na interaktywne klasyfikowanie obiektów w kosmosie - **GWIAZDA (STAR)**, **GALAKTYKA (GALAXY)**, **KWAZAR (QSA)**. Jest to fundamentalny 
    podział kosmicznych obiektów, oparty na ich charakterystyce widmowej.
    
    Klasyfikacja odbywa na na podstawie sześciu parametrów:
    - `redshift` - przesunięcie ku czerwieni, jest to zjawisko, w którym światło odległych obiektów jest przesunięte w stronę czerwonej części widma. Wynika to z rozszerzania się Wszechświata, oddalania się obiektu lub silnej grawitacji.
    - `ultraviolet` - filtr przepuszczający głównie promieniowanie ultrafioletowe, czyli krótsze fale niż światło widzialne, poniżej zakresu fioletu.
    - `green` - filtr przepuszczający światło w zakresie zielonym, w środku widzialnego widma.
    - `red` - filtr dla czerwonej części widma, dłuższe fale światła widzialnego.
    - `near infrared` - filtr w zakresie bliskiej podczerwieni, tuż za czerwonym krańcem widma widzialnego.
    - `infrared` - filtr dla jeszcze dłuższych fal, głębiej w podczerwieni niż `near infrared`.
    
    Powyższe filtry pozwalają mierzyć jasność obiektów w różnych częściach widma i dzięki temu badać ich temperaturę, skład, odległość i inne właściwości.
    
    W ramach raportu utworzone zostały dwie sekcje umożliwiające użytkownikowi interaktywną eksploracje danych oraz tworzenie interaktywnych predykcji modelu.
    
    
    Dane pochadzą z: [Star Classification - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data)
    """)

# -------------------
# 2. EKSPLORACJA
# -------------------

elif page == "Eksploracja danych":
    st.title("Eksploracja danych")

    st.markdown(f"""
    Sekcja ta składa się z wykresów, dzięki którym użytkownik może filtrować i eksplorować dane w sposób interaktywny. Zawarte w sekcji wykresy:
    - **macierz korelacji** parametrów modelu (nie interaktywne, na całym zbiorze danych),
    - **wykres PC1, PC2** po redukcji wymiarów (można wybierać co wyświetlić na wykresie, na całym zbiorze danych),
    - **histogram** (liczba obiektów w poszczególnych klasach dla przefiltrowanych danych),
    - **6 wykresów typu boxplot** (rozkład każdego z parametrów w poszczególnych klasach, na danych przefiltrowanych)
    """)

    # filtrowanie danych
    st.sidebar.header("Filtry danych")

    redshift_min, redshift_max = st.sidebar.slider("Przesunięcie ku czerwieni (redshift)", df_cleaned['redshift'].min(), 2.0, (0.0, 1.0), step=0.001)
    u_min, u_max = st.sidebar.slider("Ultrafiolet", df_cleaned['u'].min(), df['u'].max(), (15.0, 20.0), step=0.1)
    g_min, g_max = st.sidebar.slider("Światło zielone", df_cleaned['g'].min(), df['g'].max(), (15.0, 20.0), step=0.1)
    r_min, r_max = st.sidebar.slider("Światło czerwone", df_cleaned['r'].min(), df['r'].max(), (15.0, 20.0), step=0.1)
    i_min, i_max = st.sidebar.slider("Bliska podczerwień", df_cleaned['i'].min(), df['i'].max(), (15.0, 20.0), step=0.1)
    z_min, z_max = st.sidebar.slider("Podczerwień", df_cleaned['z'].min(), df['z'].max(), (15.0, 20.0), step=0.1)

    mask = (
        (df_cleaned['redshift'] >= redshift_min) & (df_cleaned['redshift'] <= redshift_max) &
        (df_cleaned['u'] >= u_min) & (df_cleaned['u'] <= u_max) &
        (df_cleaned['g'] >= g_min) & (df_cleaned['g'] <= g_max) &
        (df_cleaned['r'] >= r_min) & (df_cleaned['r'] <= r_max) &
        (df_cleaned['i'] >= i_min) & (df_cleaned['i'] <= i_max) &
        (df_cleaned['z'] >= z_min) & (df_cleaned['z'] <= z_max))

    filtered = df_cleaned.loc[
        mask,
        ['redshift', 'u', 'g', 'r', 'i', 'z', 'class']
    ]

    # zmiana nazw do wyświetlenia na macierzy korelacji
    renamed = {
        'u': 'ultraviolet',
        'g': 'green',
        'r': 'red',
        'i': 'n_infrared',
        'z': 'infrared'
    }

    # macierz korelacji
    st.markdown("#### Macierz korelacji parametrów modelu")
    corr = df_cleaned.corr().round(2)

    st.plotly_chart(px.imshow(
        corr.rename(index=renamed, columns=renamed),
        text_auto=True,
        color_continuous_scale='YlOrRd',
    ), use_container_width=True
    )

    # redukcja wymiarów PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_cleaned[['redshift', 'u', 'g', 'r', 'i', 'z']])

    df_pca = df_cleaned.copy()
    df_pca['PC1'] = components[:, 0]
    df_pca['PC2'] = components[:, 1]

    all_classes = ['STAR', 'GALAXY', 'QSO']
    selected_classes = st.multiselect('Wybierz klasy do wyświetlenia', options=all_classes)

    df_pca_filtered = df_pca[df_pca['class'].isin(selected_classes)]

    st.markdown("#### Redukcja wymiarów PCA — wizualizacja klas")
    st.plotly_chart(px.scatter(
        df_pca_filtered,
        x='PC1',
        y='PC2',
        color='class',
        labels={'class': 'Klasa'},
        color_discrete_map={
            'STAR': 'red',
            'GALAXY': 'blue',
            'QSO': 'yellow'
        },
    ),
        use_container_width=True
    )

    # histogram dla przefiltrowanych danych
    st.markdown("#### Liczba obiektów spełniające wybrane kryteria dla poszczególnych klas")
    st.markdown(f"""Liczba wszystkich obiektów spełniających kryteria: **{len(filtered)}**""")
    st.plotly_chart(px.histogram(
        filtered,
        x='class',
        color='class',
        nbins=40,
        labels={'class': 'Klasy'},
        opacity=0.7,
        color_discrete_map={
            'STAR': 'red',
            'GALAXY': 'blue',
            'QSO': 'yellow'
        },
        text_auto=True
    ),
        use_container_width=True
    )

    # wykresy typu boxplot
    st.markdown("#### Wykresy boxplot przedstawiające rozkłady parametrów w każdej z klas")
    st.plotly_chart(px.box(
        filtered,
        x='class',
        y='redshift',
        title='Rozkład redshift w poszczególnych klasach'
    ))

    st.plotly_chart(px.box(
        filtered,
        x='class',
        y='u',
        labels={'u': 'Wartość ultrafioletu'},
        title='Rozkład światła ultrafioletowego w poszczególnych klasach'

    ))

    st.plotly_chart(px.box(
        filtered,
        x='class',
        y='g',
        labels={'g': 'Wartość światła zielonego'},
        title='Rozkład światła zielonego w poszczególnych klasach'
    ))

    st.plotly_chart(px.box(
        filtered,
        x='class',
        y='r',
        labels={'r': 'Wartość światła czerwonego'},
        title='Rozkład światła czerwonego w poszczególnych klasach'
    ))

    st.plotly_chart(px.box(
        filtered,
        x='class',
        y='i',
        labels={'i': 'Wartość światła bliskiego podczerwieni'},
        title='Rozkład światła bliskiego podczerwieni w poszczególnych klasach'
    ))

    st.plotly_chart(px.box(
        filtered,
        x='class',
        y='z',
        labels={'z': 'Wartość światła podczerwonego'},
        title='Rozkład światła podczerwonego w poszczególnych klasach'
    ))


# -------------------
# 3. MODEL
# -------------------

elif page == "Model":
    st.title("Predykcja obiektu w kosmosie (klasyfikacja)")

    st.markdown("""
    Budujemy model regresji logistycznej (**LogisticRegression**) na podstawie parametrów:
    - `redshift` - przesunięcie ku czerwieni,
    - `u` - ultrafiolet,
    - `g` - światło zielone (widzialne),
    - `r` - światło czerwone (widzialne),
    - `i` - bliska podczerwień,
    - `z` - podczerwień.
    """)

    X = df_cleaned.drop(columns='class')
    y = df_cleaned['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown(f"""
    **Wyniki modelu:**
    - f1 score: `{f1_score(y_test, y_pred, average='macro'):.3f}`
    - accuracy: `{accuracy_score(y_test, y_pred):.3f}`
    """)

    # macierz pomyłek
    st.subheader("Macierz pomyłek dla każdej z klas:")
    cm = confusion_matrix(y_test, pd.DataFrame(y_pred))
    class_labels= ['GALAXY', 'QSO', 'STAR']
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    st.plotly_chart(px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Przewidziane", y="Rzeczywiste", color="Liczba"),
        x=class_labels,
        y=class_labels,),
        use_container_width=True
    )

    st.subheader("Ustaw slidery na wybranych pozycjach i uzyskaj predykcje: ")
    redshift_input = st.sidebar.slider("Przesunięcie ku czerwieni", -1.0, 7.0, 1.5, step=0.001)
    uv_input = st.sidebar.slider("Ultrafiolet", 0.0, 30.0, 15.0, step=0.001)
    green_input = st.sidebar.slider("Światło zielone", 0.0, 30.0, 15.0, step=0.001)
    red_input = st.sidebar.slider("Światło czerwone", 0.0, 30.0, 15.0, step=0.001)
    near_infra_input = st.sidebar.slider("Bliska podczerwień", 0.0, 30.0, 15.0, step=0.001)
    infra_input = st.sidebar.slider("Podczerwień", 0.0, 30.0, 15.0, step=0.001)

    # predykcja na danych ustawionych przez użytkownika
    pred_star = model.predict([[redshift_input, uv_input, green_input, red_input, near_infra_input, infra_input]])[0]
    st.success(f"Prognozowany obiekt: {pred_star}")
    if pred_star == "STAR":
        st.image("https://ocdn.eu/pulscms-transforms/1/-tpk9kpTURBXy9hMDE0OWFjODI1OWRmNDJiODYyYzAwY2E1ZWJmNGY4OC5qcGeTlQMAzQGGzRdyzQ0xlQLNBLAAwsOVAgDNAu7Cw94AA6EwAaExAaEzww",
            caption="GWIAZDA", use_container_width=True)
    elif pred_star == "GALAXY":
        st.image("https://bi.im-g.pl/im/b9/d5/16/z23941561AMP,Droga-Mleczna.jpg",
            caption="GALAKTYKA", use_container_width=True)
    else:
        st.image("https://scroll.morele.net/wp-content/uploads/2021/07/kwazar-obraz-glowny.png",
            caption="KWAZAR", use_container_width=True)
