import os
import re
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pdfplumber
from sklearn.ensemble import IsolationForest
from werkzeug.utils import secure_filename
from fpdf import FPDF  
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

def allowed_file(filename):
    """Ensure that the file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_csv_from_pdf(pdf_path):
    """Extract and parse financial data from PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text()

        lines = re.split(r'\r?\n|\r', all_text)
        lines = [line.strip() for line in lines if line.strip()]

        column_names = ['Year', 'Company', 'Market_Cap_in_B_USD', 'Revenue', 'Gross_Profit', 'Net_Income', 
                        'Cash_Flow_from_Operating', 'Cash_Flow_from_Investing', 'Cash_Flow_from_Financial_Activities', 
                        'Debt_Equity_Ratio']
        
        data_lines = [line for line in lines if line]
        data_lines = [line.split(',') for line in data_lines]
        num_cols = len(column_names)
        final_data = []

        for line in data_lines:
            cleaned_line = [element.strip() for element in line]
            if len(cleaned_line) < num_cols:
                cleaned_line += [''] * (num_cols - len(cleaned_line))
            final_data.append(cleaned_line)

        df = pd.DataFrame(final_data, columns=column_names)
        for col in column_names:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        
        df = df.dropna(thresh=len(df.columns) // 2)  

        return df

    except Exception as e:
        raise ValueError(f"An error occurred during PDF processing: {str(e)}")

def detect_anomalies(df, feature_columns):
    """Detect anomalies using Isolation Forest."""
    if df.empty:
        return pd.DataFrame()
    try:
        model = IsolationForest(contamination=0.1, random_state=42)
        df_filled = df[feature_columns].fillna(df[feature_columns].median())  
        df['Anomaly'] = model.fit_predict(df_filled)
        anomalies_df = df[df['Anomaly'] == -1].drop(columns=['Anomaly'])
        return anomalies_df
    except Exception as e:
        raise ValueError(f"Anomaly detection failed: {str(e)}")

def generate_report(metrics, anomalies, file_path):
    """Generate a PDF summary report."""
    pdf = FPDF()
    pdf.add_page()

    
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Financial Data Analysis Report", ln=True, align='C')

    
    pdf.ln(10) 
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Key Financial Metrics:", ln=True)
    for metric in metrics:
        pdf.cell(200, 10, txt=f"Year: {metric['Year']} - Revenue: {metric['Revenue']} - Net Income: {metric['Net_Income']}", ln=True)

    
    pdf.ln(10)  
    pdf.cell(200, 10, txt="Detected Anomalies:", ln=True)
    for anomaly in anomalies:
        pdf.cell(200, 10, txt=f"Year: {anomaly['Year']} - Revenue: {anomaly['Revenue']} - Net Income: {anomaly['Net_Income']}", ln=True)

    
    pdf.output(file_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main page for uploading the PDF."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the file upload and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            
            df = extract_csv_from_pdf(file_path)

           
            column_mapping = {
                'Year': 'Year',
                'Revenue': 'Revenue',
                'Net Income': 'Net_Income',
                'Cash Flow from Operating': 'Cash_Flow_from_Operating',
                'Debt/Equity Ratio': 'Debt_Equity_Ratio',
            }
            df.rename(columns=column_mapping, errors='ignore', inplace=True)

            
            metrics_columns = ['Revenue', 'Net_Income', 'Cash_Flow_from_Operating', 'Debt_Equity_Ratio']
            available_columns = [col for col in metrics_columns if col in df.columns]

            if not available_columns:
                raise ValueError(f"No required columns found. Extracted columns: {', '.join(df.columns)}")

            anomalies = detect_anomalies(df, available_columns)

            
            def nan_to_null(x):
                if isinstance(x, float) and pd.isna(x):
                    return None
                return x

            metrics = df.drop(columns=['Anomaly']).applymap(nan_to_null).to_dict(orient='records')
            anomalies = anomalies.applymap(nan_to_null).to_dict(orient='records')

            
            report_filename = os.path.splitext(filename)[0] + "_report.pdf"
            report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
            generate_report(metrics, anomalies, report_path)

            response = jsonify({'metrics': metrics, 'anomalies': anomalies, 'columns': available_columns, 'report': report_filename})
            response.headers.add('Access-Control-Allow-Origin', '*')  
            return response, 200

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Allowed file type is pdf'}), 400

if __name__ == '__main__':
    app.run(debug=True)
