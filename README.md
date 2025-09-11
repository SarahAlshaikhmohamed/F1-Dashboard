# F1-Dashboard
A simple interactive dashboard built as part of the Data Science and Machine Learning Bootcamp at Tuwaiq Academy  - Week One Project.
The project explores historical Formula 1 race winners (1950–2025), performs exploratory data analysis (EDA), and builds a Streamlit app integrated with FastAPI for predictions.

---
### Project Objective
Following week one project Instructions:
- Select a real-world dataset from Kaggle.
- Perform basic EDA (data exploration and cleaning).
- Create a dashboard using Streamlit (with optional FastAPI integration).
- Share results and presentation.

---
### Tech Stack
- Python (pandas, matplotlib, seaborn, scikit-learn)
- Streamlit (dashboard visualization)
- FastAPI (backend model serving)
- Scikit-learn (model training)

---
### How to Run
1. Clone Repository
   ``` bash
   git clone https://github.com/SarahAlshaikhmohamed/F1-Dashboard.git
   cd F1-Dashboard
   ```
2. Install Requirements
   ```bash
   pip install -r requirements.txt
   ```
3. Run FastAPI Service
   ``` bash
   uvicorn model_interface:app --reload
   ```
4. Run Streamlit Dashboard
   ```bash
   streamlit run dashboard.py
   ```

---
### Presentation
Project presentation can be found [here](https://www.canva.com/design/DAGyr6QF8LY/Vj06lWs3BIn98OPAckbREw/edit?utm_content=DAGyr6QF8LY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---
### Team Members
- Sarah Alshaikhmohamed
- Nouf Almutiri
- Abdulrahman Atar
- Norah Bindaham

---
### References
- Kaggle Dataset – [Formula 1 Grand Prix Winners (1950–2025)](https://www.kaggle.com/datasets/julianbloise/winners-formula-1-1950-to-2025?resource=download)
