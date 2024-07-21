import streamlit as st


# page conf
home = st.Page(
    "Home.py",
    title="Home",
    icon=":material/home:"
)
# Head
st.title("Dashboard SANTANA (Sistem Analisis Teks Bencana Alam)")
st.divider()
# Body
st.subheader("Anggota Kelompok:")
list_anggota = ['Mohammad Zaghy Zalayetha Sofjan',
                "Diva Restu Anggara Putra", "Wira Yudha Mahardhika"]

for anggota_index in range(len(list_anggota)):
    st.markdown(f"{anggota_index+1}. {list_anggota[anggota_index]}")

    # # Streamlit app
    # st.title("HTTP Request with Streamlit")

    # url = st.text_input("Enter the API URL:")

    # if st.button("Fetch Data"):
    #     if url:
    #         data = apiService.fetch_data("http://127.0.0.1:5000/rule?text=halo")
    #         if data:
    #             st.write("Data fetched successfully:")
    #             st.json(data)
    #     else:
    #         st.warning("Please enter a valid URL.")
