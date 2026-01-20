<<<<<<< HEAD
import os
import streamlit as st
import pandas as pd

=======
import streamlit as st
import pandas as pd
>>>>>>> c2125305f80ac929422e5245cd607e484e45cf58
from data_processor import fetch_data, preprocess_data, calculate_rfm_metrics
from rfm_analyzer import calculate_rfm_scores, prepare_for_clustering
from clustering_engine import apply_clustering, assign_cluster_names
from visualization import (
    generate_cluster_profiles,
    plot_segmentation_distribution,
    plot_rfm_comparison,
    plot_segment_sizes,
<<<<<<< HEAD
    plot_segment_revenue_percentage,
    plot_behavior_changes,   # keep if implemented
)

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


def get_segmented_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Return core segmented customer columns."""
    base_columns = [
        "CustomerID", "Recency", "Frequency", "Monetary",
        "R_Score", "F_Score", "M_Score", "RFM_Score",
        "Cluster", "Cluster_Name",
    ]
    return rfm_df[base_columns]

=======
    plot_segment_revenue_percentage, plot_behavior_changes
)
import openai
from dotenv import load_dotenv
load_dotenv(override=True)
import os
>>>>>>> c2125305f80ac929422e5245cd607e484e45cf58

def main():
    st.set_page_config(
        page_title="Bank Trust RFM Analysis",
        page_icon="💳",
        layout="wide",
<<<<<<< HEAD
        initial_sidebar_state="expanded",
    )

    st.title("💳 Bank Trust Customer RFM Analysis")
    st.markdown(
        """
        This dashboard automatically performs RFM analysis, clustering, and visualization
        for Bank Trust customers based on their transaction data.
        """
    )

    # --- Data loading (you can replace with file_uploader if needed) ---
    csv_path = "Data/Bank_Trust_Data.csv"
    raw_data = fetch_data(csv_path)

    if raw_data is None:
        st.error(f"Could not load data from: {csv_path}")
        return

    processed_data = preprocess_data(raw_data)
    if processed_data is None or processed_data.empty:
        st.error("Preprocessing failed or returned empty data.")
        return

    # --- RFM pipeline ---
    rfm_data = calculate_rfm_metrics(processed_data)
    if rfm_data is None or rfm_data.empty:
        st.error("RFM metrics calculation failed or returned empty data.")
        return

    rfm_data = calculate_rfm_scores(rfm_data)
    rfm_scaled_df = prepare_for_clustering(rfm_data)

    # --- Clustering ---
    cluster_analysis, rfm_labeled, _ = apply_clustering(rfm_scaled_df, rfm_data)
    if cluster_analysis is None or rfm_labeled is None:
        st.error("Clustering failed.")
        return

    clustered_data = assign_cluster_names(rfm_labeled, cluster_analysis)
    cluster_profiles = generate_cluster_profiles(clustered_data)

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(
        ["📊 RFM Visualization", "📋 Customer category", "💡 Recommendation"]
    )

    # ===== Tab 1: Visualizations =====
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Segmentation distribution")
            st.pyplot(plot_segmentation_distribution(clustered_data))

            st.subheader("RFM comparison by cluster")
            st.pyplot(plot_rfm_comparison(cluster_profiles))

        with col2:
            st.subheader("Segment sizes")
            st.pyplot(plot_segment_sizes(cluster_profiles))

            st.subheader("Revenue contribution by segment")
            st.pyplot(plot_segment_revenue_percentage(clustered_data))

            # Optional: if plot_behavior_changes exists
            try:
                st.subheader("Behavior changes")
                st.pyplot(plot_behavior_changes(clustered_data))
            except Exception:
                pass

    # ===== Tab 2: Segmented customers =====
    with tab2:
        st.subheader("📋 Segmented Customer Data")
        segmented_df = get_segmented_customers(clustered_data)
        st.dataframe(segmented_df, use_container_width=True)

        st.download_button(
            label="Download Segmented Customer Data as CSV",
            data=segmented_df.to_csv(index=False).encode("utf-8"),
            file_name="segmented_customers.csv",
            mime="text/csv",
        )

    # ===== Tab 3: Recommendations (LLM) =====
    with tab3:
        st.subheader("💡 Recommendations")

        api_key = os.getenv("api_key")
        if not api_key:
            st.error("API key not found in environment variable 'api_key'.")
            return

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        prompt = f"""
        You are a marketing and customer analytics expert.

        Based on this RFM cluster profile data:
        {cluster_profiles.to_dict()}

        Provide clear, actionable recommendations to:
        - Increase revenue
        - Improve customer retention
        - Boost engagement

        Structure your answer in bullet points with short explanations.
        """

        try:
            response = client.responses.create(
                model="openai/gpt-oss-120b",
                input=prompt,
            )
            # Adjust this depending on the exact response schema of the client
            try:
                text = response.output_text
            except AttributeError:
                # Fallback for other response formats
                text = str(response)

            st.markdown(text)

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

=======
        initial_sidebar_state="expanded"
    )

    st.title("💳 Bank Trust Customer RFM Analysis")
    st.markdown("""
    This dashboard automatically performs RFM analysis, clustering, and visualization
    for Bank Trust customers based on their transaction data.
    """)

    # Run the analysis pipeline
    raw_data = fetch_data("Data/Bank_Trust_Data.csv")
    processed_data = preprocess_data(raw_data)

    rfm_data = calculate_rfm_metrics(processed_data)

    # Calculate RFM scores
    rfm_data = calculate_rfm_scores(rfm_data)

    # FIX: send only numeric columns (exactly like Colab)
    rfm_scaled_df = prepare_for_clustering(rfm_data)

    # Apply Clustering
    cluster_analysis, rfm_data, _ = apply_clustering(rfm_scaled_df, rfm_data)
    clestered_data = assign_cluster_names(rfm_data, cluster_analysis)

    cluster_profiles = generate_cluster_profiles(clustered_data)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 RFM Visualization", "📊 Customer category", "Recommendation"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:    st.pyplot(plot_segmentation_distribution(clustered_data))
        st.pyplot(plot_rfm_comparison(cluster_profiles))

        with col2:
            st.pyplot(plot_segment_sizes(cluster_profiles))
            st.pyplot(plot_segment_revenue_percentage(clustered_data))

    with tab2:
        def get_segmented_customers(rfm_df):
            base_columns = [
            'CustomerID', 'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
            'Cluster', 'Cluster_Name'
        ]
        return rfm_df[base_columns]

    st.subheader("📋 Segmented Customer Data")
    segmented_df = get_segmented_customers(clustered_data)
    st.dataframe(segmented_df)

    st.download_button(
        label="Download Segmented Customer Data as CSV",
        data=segmented_df.to_csv(index=False).encode('utf-8'),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )



    # Apply clustering
    cluster_analysis, rfm_data, _ = apply_clustering(rfm_scaled_df, rfm_data)
    clustered_data = assign_cluster_names(rfm_data, cluster_analysis)

    cluster_profiles = generate_cluster_profiles(clustered_data)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 RFM Visualization", "👥 Customer Category", "📝 Recommendation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_segmentation_distribution(clustered_data))
            st.pyplot(plot_rfm_comparison(cluster_profiles))
        with col2:
            st.pyplot(plot_segment_sizes(cluster_profiles))
            st.pyplot(plot_segment_revenue_percentage(clustered_data))

        behavior_fig = plot_behavior_changes(clustered_data)
        if behavior_fig:
            st.pyplot(behavior_fig)


    with tab2:
        def get_segmented_customers(rfm_df):
            base_columns = [
                'CustomerID', 'Recency', 'Frequency', 'Monetary',
                'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
                'Cluster', 'Cluster_Name'
            ]
            available_columns = [col for col in base_columns if col in rfm_df.columns]
            segmented_customers = rfm_df[available_columns].copy()
            segmented_customers.sort_values(by=['Cluster_Name', 'Monetary'], ascending=[True, False], inplace=True)
            segmented_customers.reset_index(drop=True, inplace=True)
            return segmented_customers

    st.subheader("📋 Segmented Customer Data")
    segmented_df = get_segmented_customers(clustered_data)
    st.dataframe(segmented_df)

    csv_data = segmented_df.to_csv(index=False)
    st.download_button(
        label="Download Segmented Customer Data as CSV",
        data=segmented_df.to_csv(index=False).encode('utf-8'),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )


    with tab3:
        st.subheader("💡 Recommendations")

        prompt = f"""
        Based on this RFM cluster data:
        {cluster_profiles.to_dict()}

        Give recommendations to improve revenue, customer retention, and engagement.
    """

        client = openai.OpenAI(
            api_key=os.getenv("api_key"),
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.responses.create(
            model="openai/gpt-oss-120b",
            input=prompt,
        )

        st.markdown(response.output_text)
>>>>>>> c2125305f80ac929422e5245cd607e484e45cf58

if __name__ == "__main__":
    main()