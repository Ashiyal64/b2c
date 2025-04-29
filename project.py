import streamlit as st
import os
import google.generativeai as genai
import httpx


from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from streamlit_option_menu import option_menu
from sklearn.linear_model import  LinearRegression
import plotly.express as px
from streamlit_option_menu import option_menu
from google import genai


df=pd.read_csv("custom_dataset.csv")
#customerdata
data=pd.read_csv("Customer Data.csv")
data1=pd.read_csv("Mall_Customers.csv")


# option= [{"icon":"bi","label":"Home"},
#          {"icon":"b","label":"Work"}]
# hc.option_bar(
#     option_definition=option,
#     horizontal_orientation=True)

# Sidebar option menu

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

# --------------- Handle Query Parameters -----------------
query_params = st.query_params
if "auth" in query_params and query_params["auth"] == "True":
    st.session_state.authenticated = True
    st.session_state.user_id = query_params.get("user", "")

# --------------- Ensure User File Exists -----------------
USER_DATA_FILE = "user.csv"
if not os.path.exists(USER_DATA_FILE):
    pd.DataFrame(columns=["Email", "Pass"]).to_csv(USER_DATA_FILE, index=False)

# --------------- AUTHENTICATION SECTION -----------------
if not st.session_state.authenticated:
    st.sidebar.title("NextGenlytics")
    selected2 = st.sidebar.selectbox("LogIn or SignUp", ["LOG IN", "SIGN UP"])

    if selected2 == "SIGN UP":
        st.header("Sign Up")
        with st.form(key="signup_form"):
            email1 = st.text_input("Enter the E-MAIL")
            password1 = st.text_input("Enter the PASSWORD", type="password")
            if st.form_submit_button("SIGN UP"):
                user_file = pd.read_csv(USER_DATA_FILE)
                if email1 in user_file["Email"].values:
                    st.error("Email already exists! Please log in.")
                else:
                    new_user = pd.DataFrame([[email1, password1]], columns=["Email", "Pass"])
                    new_user.to_csv(USER_DATA_FILE, mode='a', header=False, index=False)
                    st.success("Successfully signed up! Please log in.")

    if selected2 == "LOG IN":
        st.header("Login")
        with st.form(key="login_form"):
            email = st.text_input("Enter Email")
            password = st.text_input("Enter Password", type="password")
            submitbtn = st.form_submit_button(label="Login")
            if submitbtn:
                user_file = pd.read_csv(USER_DATA_FILE)

                matched_user = user_file[user_file["Email"] == email]

                if not matched_user.empty:
                    saved_password = str(matched_user.iloc[0]["Pass"]).strip()
                    if password == saved_password:
                        st.session_state.authenticated = True
                        st.session_state.user_id = email
                        st.query_params.update(auth=True, user=email)
                        st.success("Login Successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password!")
                else:
                    st.error("Email does not exist! Please sign up first.")

if st.session_state.authenticated:

    with st.sidebar:
        selected = option_menu(
            menu_title="NextGenlytics",
            options= ["ABOUT US","🏠 Project Overview", "📊 Segmentation Dashboard", "📊 Prediction Dashboard", "📈 Analytics","🤖 CHATBOT","LogOut"],
            icons=["🏠 ","📊", "🔮", "📈    ","🤖"],
            menu_icon="cast",
            default_index=0

        )
    if selected=="🏠 Project Overview":
        st.markdown("""
            <h1 style='text-align: left; color: #0e76a8;'>
                📊 Project Overview
            </h1>
        """, unsafe_allow_html=True)
        # Apply custom styles


        # Use HTML to apply CSS classes

        st.markdown("<h2 class='section-header'>Project Summary</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='section-text'>Intelligent Customer Segmentation and Predictive Insights for B2C Growth</p>",
            unsafe_allow_html=True
        )

        st.markdown("<h2 class='section-header'>Objective</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='section-text'>The objective of this project is to analyze customers, segment them into distinct groups, and predict their future actions to enable personalized marketing strategies and improved customer engagement.</p>",
            unsafe_allow_html=True
        )

        st.markdown("<h2 class='section-header'>Overview</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='section-text'>In modern B2C businesses, customer understanding is critical for growth. This project leverages machine learning to segment customers based on demographics and purchasing behavior. A future extension will include predictive analytics to forecast customer purchases, churn, and recommend tailored offers.</p>",
            unsafe_allow_html=True,
        )

    if selected=="ABOUT US":

        st.subheader("Welcome to NextGenlytics")

        st.write("""
        At *NextGenlytics*, we are passionate about turning data into actionable intelligence. 
        As a forward-thinking tech venture, we specialize in harnessing the power of data analytics and machine learning 
        to solve real-world business challenges. Our mission is to empower B2C businesses with intelligent solutions 
        that enhance customer understanding, boost engagement, and drive sustainable growth.
        """)

        st.subheader("Who We Are")

        st.write("""
        We are a dedicated team of data scientists, developers, and innovators driven by curiosity and a deep desire to 
        make data meaningful. Our flagship project, *"Intelligent Customer Segmentation and Predictive Insights for B2C Growth,"* 
        reflects our core belief — that behind every dataset lies a story waiting to be told, and every story has the power 
        to transform a business.
        """)
        st.subheader("What We Do")

        st.write("""
              In today’s competitive B2C landscape, personalized experiences are no longer optional — they’re expected. 
              At NextGenlytics, we develop systems that:
              - Segment customers using advanced clustering and profiling techniques  
              - Predict customer behavior and preferences using machine learning models  
              - Provide actionable insights for personalized marketing strategies  
              - Help businesses build lasting relationships with their customers

              By leveraging cutting-edge technologies and data-driven approaches, we bridge the gap between data and decision-making.
              """)

        st.subheader("Our Vision")

        st.write("""
              To become a trusted partner for B2C businesses by delivering intelligent, scalable, and user-centric solutions 
              that inspire smarter business strategies and deeper customer connections.
              """)

        st.subheader("Our Values")

        st.write("""
              - *Innovation:* We constantly explore new ideas and technologies to stay ahead.  
              - *Integrity:* We believe in transparency, ethics, and delivering what we promise.  
              - *Impact:* Our solutions are designed to create measurable and meaningful business value.  
              - *Collaboration:* We grow through shared ideas and teamwork — both within our team and with our partners.
              """)

    if selected=="📊 Segmentation Dashboard":
        st.markdown("""
            <style>
                h1, h2 {
                    text-align: left;
                    color: #0e76a8;
                    margin-bottom: 10px;
                }
    
                .step-header {
                    font-size: 22px;
                    font-weight: 600;
                    color: #444;
                    margin-top: 30px;
                    margin-bottom: 10px;
                }
    
                .bi {
                    margin-right: 8px;
                }
    
                /* Optional: style slider label */
                .stSlider > label {
                    font-weight: 500;
                    color: #333;
                }
            </style>
    
            <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
        """, unsafe_allow_html=True)

        # Title
        st.markdown("<h1><i class='bi bi-graph-up'></i> Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)

        # Step 1 - Load Data
        st.markdown("<h2 class='step-header'><i class='bi bi-upload'></i> Step 1: Load Data</h2>", unsafe_allow_html=True)
        st.dataframe(df)

        # Step 2 - Clustering
        st.markdown("<h2 class='step-header'><i class='bi bi-diagram-3'></i> Step 2: Perform Clustering</h2>",
                    unsafe_allow_html=True)

        degree = st.slider("Please Enter the number of Clusters", 1, 5, 2)

        # Apply clustering
        x = df[['Age', 'Income', 'Score']]
        kmeans = KMeans(n_clusters=degree, n_init=10)
        df['cluster'] = kmeans.fit_predict(x)

        # Show updated dataframe with cluster labels
        st.dataframe(df)

        # Color map for up to 5 clusters
        color_map = {
            0: 'red',
            1: 'green',
            2: 'pink',
            3: 'yellow',
            4: 'black'
        }

        # Step 3 - Visualisation
        st.markdown(
            "<h2 class='step-header'><i class='bi bi-scatter-chart'></i> Step 3: Cluster Visualisation</h2>",
            unsafe_allow_html=True
        )

        # Plot with custom colors and cluster labels as text
        fig = px.scatter(
            df,
            x='Age',
            y='Income',
            color=df['cluster'].astype(str),  # Color by cluster
            text=df['cluster'],  # Show cluster label on each point
            color_discrete_map={str(k): v for k, v in color_map.items() if k < degree},
            hover_data=['Score', 'cluster']
        )

        fig.update_traces(textposition='top center')  # Position of the label
        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Income",
            bargap=0.1
        )

        # Show plot
        st.plotly_chart(fig, use_container_width=True)



    if selected== "📊 Prediction Dashboard":
        st.header("PREDICTION DASHBOARD")
        reg = linear_model.LinearRegression()
        reg.fit(df[["Age","Income"]],df["Score"])
        with st.form(key="form1"):
            age=st.text_input("Enter a Age")
            income=st.text_input("Enter a Income")
            submit=st.form_submit_button("CLICK")
            if submit:
                Age=float(age)
                Income=float(income)
                pre=reg.predict([[Age,Income]])
                st.success(f" predicted score:{pre[0]}")

                fig=px.scatter(df,x='Age',y='Score')
                st.header("predict score base on income and age")

                fig.add_scatter(
                    x=df['Age'],
                    y=df['Score'],
                    mode='markers+text',
                    marker=dict(color='blue', size=8),

                    name='Actual'
                )



                fig.add_scatter(x=[Age],y=[pre][0],marker=dict(color='red',size=10),name='Predicted',
                    mode='markers+text',)
                st.plotly_chart(fig, use_container_width=True)



    if selected=="📈 Analytics":
        st.markdown("""
            <style>
                .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                    text-align: left;
                }
                h1, h2, h3, h4 {
                    color: #2E86C1;
                }
                .stTextInput, .stNumberInput {
                    border-radius: 8px;
                    padding: 5px;
                }
                .stButton > button {
                    background-color: #3498DB;
                    color: white;
                    border-radius: 8px;
                    padding: 10px 24px;
                }
                .stButton > button:hover {
                    background-color: #2E86C1;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("## 📈 Welcome To Customer Analytics")
        st.markdown("###  Customer Mall Analysis")
        st.markdown("### 📅 Annual Income (k$) vs Spending Score (1-100)")

        kmeans = KMeans(n_clusters=3, n_init=10)
        data1["cluster"] = kmeans.fit_predict(data1[["Annual Income (k$)", "Spending Score (1-100)", "Age"]])

        fig = px.scatter(data1, x="Annual Income (k$)", y="Spending Score (1-100)", color="cluster",
                         title="Annual Income (k$) vs Spending Score (1-100)", color_continuous_scale="viridis")
        fig.update_layout(xaxis_title="Annual Income (k$)", yaxis_title="Spending Score")
        st.plotly_chart(fig)

        with st.form(key="form2"):
            st.markdown("### 📉 Income-wise Spending Cluster")
            Age = st.text_input("Enter Age")
            spendigscore = st.number_input("Enter Spending Score")
            anualincome = st.number_input("Enter Annual Income")
            submit = st.form_submit_button(label="🔎 Predict Cluster")

            if submit:
                age = float(Age)
                spendigscore = float(spendigscore)
                anualincome = float(anualincome)
                pre = kmeans.predict([[anualincome, spendigscore, age]])
                st.success(f"Predicted Cluster: {pre[0]}")
                st.info("""
                            *Cluster 0*: Low Income, Low Spending  
                            *Cluster 1*: High Income, High Spending  
                            *Cluster 2*: High Income, Low Spending  
                            *Cluster 3*: Low Income, High Spending  
                            *Cluster 4*: Moderate Income, Moderate Spending
                            """)

        # data1['Age Group'] = pd.cut(data1['Age'], bins=range(0, 201, 10), right=False)
        # grouped_df = data1.groupby('Age Group', observed=True)['Spending Score (1-100)'].mean().reset_index()
        # grouped_df['Age Group'] = grouped_df['Age Group'].astype(str)
        #
        # # Create bar chart
        # fig = px.bar(
        #     grouped_df,
        #     x="Age Group",
        #     y="Spending Score (1-100)",
        #     title='Average Spending Score by Age Group',
        #     labels={"Spending Score (1-100)": "Avg Spending Score"}
        # )
        #
        # st.plotly_chart(fig)

        st.markdown("### 📊 BAR CHART OF GENDER AND SPENDING SCORES")
        count = data1.groupby("Gender")["Spending Score (1-100)"].mean()
        st.write(count)

        fig = px.bar(count, x=["Male", "Female"], y="Spending Score (1-100)",
                     color_discrete_map={"Male": "pink", "Female": "blue"}, color=["Male", "Female"])
        fig.update_layout(xaxis_title="Gender", yaxis_title="Spending Score (1-100)", width=1000, height=500,
                          title="Gender vs Spending Score")
        fig.update_traces(width=0.3)
        st.plotly_chart(fig)

        st.markdown("### 🌐 BAR CHART OF AGE RNAGE AND SPENDING SCORES")
        bins = [0, 20, 30, 40, 50, 60, 70]
        group = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70"]
        data1["Agegroup"] = pd.cut(data1["Age"], bins=bins, labels=group, ordered=False)
        groupage = data1.groupby("Agegroup")["Spending Score (1-100)"].mean()

        fig = px.bar(groupage, x=group, y="Spending Score (1-100)",
                     color=group, barmode='group',
                     color_discrete_map={"0-20": "red", "20-30": "yellow", "30-40": "red",
                                         "40-50": "pink", "50-60": "green", "60-70": "gray"})
        fig.update_layout(xaxis_title="Age Groups", yaxis_title="Spending Score (1-100)",
                          title="Age Group vs Spending Score", bargap=0.5)
        fig.update_traces(width=0.5)
        st.plotly_chart(fig)

        st.markdown("### 🔹 Scatter: Income vs Score")
        fig = px.scatter(data1, x="Annual Income (k$)", y="Spending Score (1-100)",size="Spending Score (1-100)")
        fig.update_layout(xaxis_title="Annual Income (k$)", yaxis_title="Spending Score (1-100)",  bargap=0.1)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("### 📊 Histogram OF AGE")
        fig = px.histogram(data1, x="Age")
        fig.update_layout(
            xaxis_title='Age Range',
            yaxis_title='Frequency',

            title="Age vs Frequency",

            template='plotly_white'
        )
        st.plotly_chart(fig)


    # Chatbot Section
    if selected == "🤖 CHATBOT":
        # Ensure that the file is uploaded before processing
        fillle = st.file_uploader("Hey, upload a CSV file", type="csv")

        if fillle:
            df = pd.read_csv(fillle)
            st.dataframe(df)

            # Convert each row to a readable text line
            text_data = ""
            for index, row in df.iterrows():
                row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                text_data += row_text + "\n"

            # Optional: Save to a .txt file
            with open("converted_text.txt", "w", encoding="utf-8") as f:
                f.write(text_data)

            with open("converted_text.txt", "r", encoding="utf-8") as f:
                content = f.read()

            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Input prompt from user
            if prompt := st.chat_input("What is up?"):
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Append user prompt with the provided content
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"{content} Answer according to this data which I have provided into this text: {prompt}"
                })

                # Initialize the GenAI client
                client = genai.Client(api_key="AIzaSyCbNcbjk9_wbsKwSuwNQflPNi6SquD2CDM")

                model_name = "gemini-2.0-flash"  # or the appropriate model name
                messages = [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ]

                # Construct the prompt text for the model
                prompt_text = ""
                for msg in messages:
                    if msg["role"] == "user":
                        prompt_text += f"User: {msg['content']}\n"
                    else:
                        prompt_text += f"Assistant: {msg['content']}\n"
                prompt_text += "Assistant:"

                # Make the API call and get the response
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text
                )

                # Get the response text
                full_response = response.text  # or list of messages
                st.write(full_response)

                # Append assistant's response to the session state
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        else:
            st.warning("Please upload a CSV file to proceed.")

    if selected == "ℹ About Us":
        st.header("About Us")

        st.subheader("Welcome to NextGenlytics")

        st.write("""
                  At *NextGenlytics*, we are passionate about turning data into actionable intelligence. 
                  As a forward-thinking tech venture, we specialize in harnessing the power of data analytics and machine learning 
                  to solve real-world business challenges. Our mission is to empower B2C businesses with intelligent solutions 
                  that enhance customer understanding, boost engagement, and drive sustainable growth.
                  """)

        st.subheader("Who We Are")

        st.write("""
                  We are a dedicated team of data scientists, developers, and innovators driven by curiosity and a deep desire to 
                  make data meaningful. Our flagship project, *"Intelligent Customer Segmentation and Predictive Insights for B2C Growth,"* 
                  reflects our core belief — that behind every dataset lies a story waiting to be told, and every story has the power 
                  to transform a business.
                  """)

        st.subheader("What We Do")

        st.write("""
                  In today’s competitive B2C landscape, personalized experiences are no longer optional — they’re expected. 
                  At NextGenlytics, we develop systems that:
                  - Segment customers using advanced clustering and profiling techniques  
                  - Predict customer behavior and preferences using machine learning models  
                  - Provide actionable insights for personalized marketing strategies  
                  - Help businesses build lasting relationships with their customers

                  By leveraging cutting-edge technologies and data-driven approaches, we bridge the gap between data and decision-making.
                  """)

        st.subheader("Our Vision")

        st.write("""
                  To become a trusted partner for B2C businesses by delivering intelligent, scalable, and user-centric solutions 
                  that inspire smarter business strategies and deeper customer connections.
                  """)

        st.subheader("Our Values")

        st.write("""
                  - *Innovation:* We constantly explore new ideas and technologies to stay ahead.  
                  - *Integrity:* We believe in transparency, ethics, and delivering what we promise.  
                  - *Impact:* Our solutions are designed to create measurable and meaningful business value.  
                  - *Collaboration:* We grow through shared ideas and teamwork — both within our team and with our partners.
                  """)

    if selected == "LogOut":
             st.session_state.authenticated =False
             st.session_state.user_id = None
             st.query_params.clear()
             st.success("Logged out Successfully.......Log In again to use out website")
             st.rerun()


