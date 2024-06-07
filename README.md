**YouTube Analytics Dashboard**

This repository contains the code for a YouTube Analytics Dashboard developed using Python and Streamlit. The dashboard provides insights into YouTube video performance metrics such as views, likes, subscribers, shares, comments, RPM (Revenue per Mille), engagement ratio, and more.
Download the dataset [here](https://www.kaggle.com/datasets/kenjee/ken-jee-youtube-data)

### Setup Instructions:
1. Clone the repository to your local machine.
2. Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).
3. Navigate to the project directory using the command line.
4. Install the required dependencies by running `pip install -r requirements.txt`.
5. Run the Streamlit app using the command `streamlit run dashboard.py`.
6. Access the dashboard in your web browser at the URL provided by Streamlit.

### Features:
1. **Overview Section:**
   - Provides an overview of key metrics for all videos.
   - Displays metrics such as total views, likes, subscribers gained, average duration watched, and more.

2. **Individual Video Analysis:**
   - Allows users to select a specific video for in-depth analysis.
   - Visualizes the relationship between views and subscription status by country using a bar chart.
   - Compares cumulative views for the first 30 days since publication with percentiles of historical data.

3. **Data Processing:**
   - Merges data from multiple datasets based on common identifiers such as video ID and publish date.
   - Calculates additional metrics such as days since publication and cumulative views.

4. **Styling:**
   - Applies custom styling to the dashboard for improved readability and visual appeal.
   - Formats numeric values as percentages for better interpretation.

### Data Sources:
- The dashboard utilizes data extracted from YouTube Analytics reports.
- Data includes video metadata, audience engagement metrics, and performance statistics.

### Technologies Used:
- Python: Programming language used for data processing and dashboard development.
- Streamlit: Open-source app framework used for building interactive web applications.
- Pandas: Library used for data manipulation and analysis.
- Plotly: Library used for creating interactive visualizations.

### Contributors:
- [Ken Jee GitHub Profile](https://github.com/PlayingNumbers)
  
### Future Enhancements:
- Incorporate additional metrics and visualizations based on user feedback.
- Optimize data processing pipelines for improved performance.
- Add support for data updates and real-time analytics.
