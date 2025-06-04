import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎓 Jaya Jaya Institut</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Student Dropout Prediction System</h2>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        model = joblib.load('student_dropout_model.pkl')
        scaler = joblib.load('student_dropout_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run the training notebook first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load sample data for reference
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("student_data.csv")
        
        # Create Is_Dropout column if not exists
        if 'Is_Dropout' not in df.columns:
            df['Is_Dropout'] = (df['Status'] == 'Dropout').astype(int)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load model and data
model, scaler, feature_names = load_model()
data = load_data()

# Sidebar navigation
st.sidebar.markdown("## 🎓 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "ℹ️ About"]
)

# Main content based on selected page
if page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">🏠 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    if data is None:
        st.warning("Data not available. Please check data file.")
    else:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Students",
                value=f"{len(data):,}",
                delta=None
            )
        
        with col2:
            dropout_rate = (data['Status'] == 'Dropout').mean() * 100
            st.metric(
                label="Dropout Rate",
                value=f"{dropout_rate:.1f}%",
                delta=f"{dropout_rate-30:.1f}% vs avg"
            )
        
        with col3:
            graduate_rate = (data['Status'] == 'Graduate').mean() * 100
            st.metric(
                label="Graduate Rate",
                value=f"{graduate_rate:.1f}%",
                delta=f"{graduate_rate-50:.1f}% vs avg"
            )
        
        with col4:
            enrolled_rate = (data['Status'] == 'Enrolled').mean() * 100
            st.metric(
                label="Currently Enrolled",
                value=f"{enrolled_rate:.1f}%",
                delta=None
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Status Distribution")
            status_counts = data['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Student Status Distribution",
                color_discrete_map={
                    'Graduate': '#4CAF50',
                    'Dropout': '#F44336',
                    'Enrolled': '#2196F3'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Age Distribution")
            fig = px.histogram(
                data,
                x='Age_at_enrollment',
                color='Status',
                title="Age Distribution by Status",
                nbins=20,
                color_discrete_map={
                    'Graduate': '#4CAF50',
                    'Dropout': '#F44336',
                    'Enrolled': '#2196F3'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Academic performance analysis
        st.subheader("🎯 Academic Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # First semester performance
            fig = px.box(
                data,
                x='Status',
                y='Curricular_units_1st_sem_approved',
                title="1st Semester Approved Units by Status",
                color='Status',
                color_discrete_map={
                    'Graduate': '#4CAF50',
                    'Dropout': '#F44336',
                    'Enrolled': '#2196F3'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Admission grade performance
            fig = px.box(
                data,
                x='Status',
                y='Admission_grade',
                title="Admission Grade by Status",
                color='Status',
                color_discrete_map={
                    'Graduate': '#4CAF50',
                    'Dropout': '#F44336',
                    'Enrolled': '#2196F3'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prediction":
    st.markdown('<h2 class="sub-header">🔮 Student Dropout Prediction</h2>', unsafe_allow_html=True)
    
    # Prediction mode selection
    prediction_mode = st.selectbox(
        "Select prediction mode:",
        ["Individual Student Prediction", "Batch Prediction", "Risk Scoring"]
    )
    
    if prediction_mode == "Individual Student Prediction":
        st.subheader("👤 Individual Student Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Basic Information")
            age = st.slider("Age at Enrollment", 17, 70, 20)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            marital_status = st.selectbox("Marital Status", [1, 2, 3, 4, 5, 6], 
                                        format_func=lambda x: ["Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"][x-1])
            scholarship = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            international = st.selectbox("International Student", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            st.markdown("### 💰 Financial Information")
            debtor = st.selectbox("Debtor Status", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            tuition_up_to_date = st.selectbox("Tuition Up to Date", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            st.markdown("### 🎓 Academic Information")
            admission_grade = st.slider("Admission Grade", 95.0, 190.0, 120.0)
            prev_qual_grade = st.slider("Previous Qualification Grade", 95.0, 190.0, 120.0)
            
            units_1st_sem_enrolled = st.slider("1st Sem Units Enrolled", 0, 26, 6)
            units_1st_sem_approved = st.slider("1st Sem Units Approved", 0, 26, 5)
            units_1st_sem_grade = st.slider("1st Sem Average Grade", 0.0, 20.0, 10.0)
            
            units_2nd_sem_enrolled = st.slider("2nd Sem Units Enrolled", 0, 23, 6)
            units_2nd_sem_approved = st.slider("2nd Sem Units Approved", 0, 23, 5)
            units_2nd_sem_grade = st.slider("2nd Sem Average Grade", 0.0, 20.0, 10.0)
        
        if st.button("🔮 Predict Dropout Risk", type="primary"):
            # Create input array (simplified version - you'll need to include all 36 features)
            input_data = np.array([
                marital_status, 1, 0, 9147, 1, 1, prev_qual_grade, 1, 1, 1, 4, 4,
                admission_grade, 0, 0, debtor, tuition_up_to_date, gender, scholarship,
                age, international, 0, units_1st_sem_enrolled, units_1st_sem_enrolled,
                units_1st_sem_approved, units_1st_sem_grade, 0, 0, units_2nd_sem_enrolled,
                units_2nd_sem_enrolled, units_2nd_sem_approved, units_2nd_sem_grade, 0,
                10.8, 0.3, 2.74
            ]).reshape(1, -1)
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            dropout_prob = model.predict_proba(input_scaled)[0][1]
            prediction = "High Risk" if dropout_prob > 0.7 else "Medium Risk" if dropout_prob > 0.3 else "Low Risk"
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Dropout Probability", f"{dropout_prob:.1%}")
            
            with col2:
                st.metric("Risk Level", prediction)
            
            with col3:
                confidence = max(dropout_prob, 1-dropout_prob)
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Risk level styling
            if dropout_prob > 0.7:
                st.markdown(f"""
                <div class="risk-high">
                    <h4>⚠️ HIGH RISK STUDENT</h4>
                    <p>This student has a <strong>{dropout_prob:.1%}</strong> probability of dropping out. 
                    Immediate intervention is recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            elif dropout_prob > 0.3:
                st.markdown(f"""
                <div class="risk-medium">
                    <h4>⚡ MEDIUM RISK STUDENT</h4>
                    <p>This student has a <strong>{dropout_prob:.1%}</strong> probability of dropping out. 
                    Monitor closely and provide support.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h4>✅ LOW RISK STUDENT</h4>
                    <p>This student has a <strong>{dropout_prob:.1%}</strong> probability of dropping out. 
                    Continue with regular academic support.</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif prediction_mode == "Batch Prediction":
        st.subheader("📊 Batch Prediction")
        st.write("Upload a CSV file with student data for batch prediction.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())
                
                if st.button("🔮 Run Batch Prediction"):
                    # Ensure the data has the right features (simplified for demo)
                    if len(batch_df.columns) >= 36:
                        batch_scaled = scaler.transform(batch_df.iloc[:, :36])
                        predictions = model.predict_proba(batch_scaled)[:, 1]
                        
                        batch_df['Dropout_Probability'] = predictions
                        batch_df['Risk_Level'] = pd.cut(predictions, 
                                                      bins=[0, 0.3, 0.7, 1.0], 
                                                      labels=['Low', 'Medium', 'High'])
                        
                        st.write("### 📋 Prediction Results")
                        st.dataframe(batch_df[['Dropout_Probability', 'Risk_Level']])
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("⚠️ The uploaded file must have at least 36 columns matching the model features.")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        else:
            st.info("👆 Please upload a CSV file to proceed with batch prediction.")
    
    elif prediction_mode == "Risk Scoring":
        st.subheader("⚖️ Risk Scoring System")
        
        if data is None:
            st.warning("Data not available. Please check data file.")
        else:
            # Sample some students for risk scoring
            sample_students = data.sample(n=10, random_state=42)
            
            # Calculate risk scores
            X_sample = sample_students.drop(['Status', 'Is_Dropout'], axis=1)
            X_sample_scaled = scaler.transform(X_sample)
            risk_scores = model.predict_proba(X_sample_scaled)[:, 1]
            
            # Create risk dataframe
            risk_df = pd.DataFrame({
                'Student_ID': range(1, 11),
                'Age': sample_students['Age_at_enrollment'].values,
                'Gender': sample_students['Gender'].map({0: 'Female', 1: 'Male'}).values,
                'Admission_Grade': sample_students['Admission_grade'].values,
                'Risk_Score': risk_scores,
                'Risk_Level': pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High']),
                'Actual_Status': sample_students['Status'].values
            })
            
            st.write("### 📊 Risk Score Analysis")
            st.dataframe(risk_df.style.format({'Risk_Score': '{:.1%}', 'Admission_Grade': '{:.1f}'}))
            
            # Risk distribution chart
            fig = px.histogram(
                risk_df, 
                x='Risk_Score', 
                nbins=20,
                title="Risk Score Distribution",
                labels={'Risk_Score': 'Dropout Risk Score', 'count': 'Number of Students'}
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Analytics":
    st.markdown('<h2 class="sub-header">📊 Analytics</h2>', unsafe_allow_html=True)
    
    if data is None:
        st.warning("Data not available. Please check data file.")
    else:
        # Feature importance
        st.subheader("🎯 Most Important Features")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features for Dropout Prediction",
                labels={'Importance': 'Feature Importance Score'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Feature Correlations")
        
        # Select key numeric features for correlation
        key_features = [
            'Age_at_enrollment', 'Admission_grade', 'Previous_qualification_grade',
            'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
            'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade',
            'Is_Dropout'
        ]
        
        corr_matrix = data[key_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Key Features",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Demographics analysis
        st.subheader("👥 Demographics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender analysis
            gender_dropout = data.groupby('Gender')['Is_Dropout'].agg(['count', 'sum', 'mean']).reset_index()
            gender_dropout['Gender_Label'] = gender_dropout['Gender'].map({0: 'Female', 1: 'Male'})
            
            fig = px.bar(
                gender_dropout,
                x='Gender_Label',
                y='mean',
                title="Dropout Rate by Gender",
                labels={'mean': 'Dropout Rate', 'Gender_Label': 'Gender'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scholarship analysis
            scholarship_dropout = data.groupby('Scholarship_holder')['Is_Dropout'].agg(['count', 'sum', 'mean']).reset_index()
            scholarship_dropout['Scholarship_Label'] = scholarship_dropout['Scholarship_holder'].map({0: 'No Scholarship', 1: 'Scholarship'})
            
            fig = px.bar(
                scholarship_dropout,
                x='Scholarship_Label',
                y='mean',
                title="Dropout Rate by Scholarship Status",
                labels={'mean': 'Dropout Rate', 'Scholarship_Label': 'Scholarship Status'}
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎓 Student Dropout Prediction System
    
    This application uses machine learning to predict student dropout risk based on various academic, demographic, and socio-economic factors.
    
    ### 📊 Dataset Information
    
    The dataset contains information about **4,424 students** from a higher education institution, including:
    - **37 features** covering demographics, academic performance, and socio-economic factors
    - **Three possible outcomes**: Graduate, Dropout, or Enrolled
    - **Dropout rate**: 32.12% of students in the dataset
    
    ### 🤖 Model Information
    
    - **Algorithm**: Gradient Boosting Classifier
    - **Accuracy**: 87.8%
    - **AUC Score**: 93.07%
    - **Key Features**: Academic performance metrics are the most important predictors
    
    ### 📈 Key Insights
    
    1. **Academic Performance**: Students' performance in the first and second semesters is the strongest predictor of dropout risk
    2. **Age Factor**: Older students at enrollment tend to have higher dropout rates
    3. **Financial Status**: Students with up-to-date tuition payments have lower dropout risk
    4. **Scholarships**: Scholarship holders generally have better retention rates
    
    ### 🔍 Features Description
    
    **Demographics:**
    - Age at enrollment, Gender, Marital status, Nationality
    - Parents' education and occupation levels
    
    **Academic:**
    - Previous qualification grades, Admission grades
    - Course units enrolled, approved, and average grades per semester
    
    **Financial:**
    - Scholarship status, Debtor status, Tuition payment status
    
    **Socio-Economic:**
    - Displacement status, Special educational needs
    - Economic indicators (GDP, unemployment rate, inflation rate)
    
    ### 💡 How to Use This System
    
    1. **Dashboard**: Overview of student population and key metrics
    2. **Prediction**: 
       - Individual predictions for single students
       - Batch processing for multiple students
       - Risk scoring for cohort analysis
    3. **Analytics**: Deep dive into feature importance and correlations
    
    ### ⚠️ Important Notes
    
    - This model is trained on historical data and should be used as a support tool
    - Regular model retraining is recommended as new data becomes available
    - Consider additional factors and context when making intervention decisions
    
    ### 📞 Contact & Support
    
    For questions about this application or to report issues, please contact the development team.
    
    ---
    
    **Built with**: Streamlit, Scikit-learn, Plotly, and ❤️
    """)
    
    # Dataset sample
    st.subheader("📋 Dataset Sample")
    if data is not None:
        st.dataframe(data.head(10))
    else:
        st.warning("Data not available. Please check data file.")
    
    # Technical specifications
    with st.expander("🔧 Technical Specifications"):
        st.markdown("""
        ### Model Performance Metrics
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 87.8% |
        | AUC Score | 93.07% |
        | Precision (Dropout) | 86% |
        | Recall (Dropout) | 74% |
        | F1-Score (Dropout) | 80% |
        
        ### Feature Engineering
        
        - **Scaling**: StandardScaler for numerical features
        - **Encoding**: Label encoding for categorical variables
        - **Target**: Binary classification (Dropout vs Non-Dropout)
        
        ### Data Split
        
        - **Training**: 80% (3,539 students)
        - **Testing**: 20% (885 students)
        - **Stratification**: Maintained class balance across splits
        """)

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666666; padding: 1rem;'>
    🎓 Student Dropout Prediction System | Built with Streamlit | © 2024
</div>
""", unsafe_allow_html=True)