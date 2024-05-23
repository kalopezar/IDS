import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from joblib import dump
import base64
from io import BytesIO

# Function to generate HTML file
def generate_html_report():
    # File paths
    normal_traffic_file = "/home/seed/Documents/Capstone/IDS1/RandomForest/VEHICLE_UPDATES_NoBogusInfo_20220224-010030.csv"
    abnormal_traffic_file = "/home/seed/Documents/Capstone/IDS1/RandomForest/VEHICLE_UPDATES_BogusInfoAttack_2022_04_24-06_33_56_PM.csv"

    # Load the CSV files for normal and abnormal traffic
    normal_traffic_data = pd.read_csv(normal_traffic_file)
    abnormal_traffic_data = pd.read_csv(abnormal_traffic_file)

    # Add labels indicating normal and abnormal traffic
    normal_traffic_data['Label'] = 0  # 0 for 'Normal'
    abnormal_traffic_data['Label'] = 1  # 1 for 'Abnormal'

    # Concatenate the two datasets
    data = pd.concat([normal_traffic_data, abnormal_traffic_data], ignore_index=True)

    # Selecting features (time and speed) and target variable (label)
    X = data[['Time', 'Speed']]
    y = data['Label']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Classification Report
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    classification_df = pd.DataFrame(classification_rep).transpose()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # Dump the model
    dump(model, 'random_forest_model.joblib')

    # Select only four vehicles to plot
    vehicles_to_plot = ['veh_0', 'veh_1', 'veh_2', 'veh_3']

    # Plotting the line graph for selected vehicles in normal traffic
    plt.figure(figsize=(10, 6))
    for vehicle in vehicles_to_plot:
        vehicle_data = normal_traffic_data[normal_traffic_data['Name'] == vehicle]
        plt.plot(vehicle_data['Time'], vehicle_data['Speed'], label=vehicle)

    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Time vs Speed for Selected Vehicles (Normal Traffic)')
    plt.legend()
    plt.grid(True)
    plt.savefig('normal_traffic_graph.png')
    plt.close()

    # Plotting the line graph for selected vehicles in abnormal traffic
    plt.figure(figsize=(10, 6))
    for vehicle in vehicles_to_plot:
        vehicle_data = abnormal_traffic_data[abnormal_traffic_data['Name'] == vehicle]
        plt.plot(vehicle_data['Time'], vehicle_data['Speed'], label=vehicle)

    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Time vs Speed for Selected Vehicles (Abnormal Traffic)')
    plt.legend()
    plt.grid(True)
    plt.savefig('abnormal_traffic_graph.png')
    plt.close()

    # Convert classification report DataFrame to HTML table
    html_table = classification_df.to_html()

    # Convert confusion matrix to HTML table
    conf_matrix_html = pd.DataFrame(conf_matrix).to_html()

    # Plotting ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

    # Plotting Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.close()

    # Writing HTML to a file
    with open("analysis_report.html", "w") as f:
        f.write('<div class="container">')
        f.write('<div class="image-container">')
        f.write('<img src="head.jpg" alt="Your Image">')
        f.write('</div>')
        f.write('<br>')
        f.write('<br>')
        f.write('<br>')
        f.write("<h1>Graphs:</h1>")
        f.write("<h2>Normal Traffic:</h2>")
        f.write('<img src="normal_traffic_graph.png"><br>')
        f.write("<h2>Abnormal Traffic:</h2>")
        f.write('<img src="abnormal_traffic_graph.png"><br>')
        f.write("<h1>Classification Report:</h1>")
        f.write(html_table)
        f.write("<h1>Confusion Matrix:</h1>")
        f.write(conf_matrix_html)
        f.write("<h1>ROC Curve:</h1>")
        f.write('<img src="roc_curve.png"><br>')
        f.write("<h1>Precision-Recall Curve:</h1>")
        f.write('<img src="precision_recall_curve.png"><br>')
        f.write('</div>')

# Generate HTML report and train Random Forest model
generate_html_report()

