# Agentic AI Health Symptom Checker

## Project Overview

The **Agentic AI Health Symptom Checker** helps users understand their health conditions by analyzing symptoms described in natural language and medical images. It provides probable causes, urgency levels, preventive advice, and recommendations on when to consult a healthcare professional.

The system uses verified medical data from trusted sources such as the World Health Organization (WHO) and government health portals to ensure accuracy and reliability. It supports multi-language interaction and avoids risks of self-diagnosis by focusing on educational and referral-based guidance.

## Key Features

- Natural language symptom input (e.g., “I have a sore throat and fever”)  
- Multimodal input support with medical images  
- Probable condition prediction and urgency assessment  
- Preventive care advice and referral suggestions  
- Multilingual support  
- Based on verified medical databases to reduce misinformation  

## Technology Stack

- **IBM watsonx.ai**: AI model deployment and inference platform  
- **Pixtral 12B**: Multimodal large language model (text + image)  
- **IBM Cloud Lite & Granite**: Hosting, API services, and backend infrastructure  
- Python libraries: `ibm_watsonx_ai`, `requests`, `Pillow`, `base64`  

## Setup Instructions

1. Create an IBM Cloud Lite account and access [watsonx.ai](https://watsonx.ai) (available in Europe regions).  
2. Create a watsonx.ai project and runtime service (Lite plan).  
3. Generate API keys and set up credentials in your environment.  
4. Use the provided Jupyter Notebook to input symptoms and images.  
5. The Pixtral 12B model processes the input and returns probable conditions and advice.

## Usage

- Input symptoms as natural language text.  
- Upload or provide URLs for medical images if available.  
- Receive probable diagnoses, urgency levels, and care recommendations.  
- Supports multiple languages and provides educational suggestions.  

## Results

The model effectively interprets symptoms and images, offering accurate medical condition predictions and urgency assessments, which aid early detection and informed health decisions.

## Challenges and Future Work

- Integration of real-time data for more dynamic advice  
- Expansion of medical knowledge databases and continual updates  
- Enhanced multilingual support and user experience  
- Exploration of edge computing for faster, local processing  

## Acknowledgments

This internship project is sponsored by **Edunet Foundation** in partnership with **IBM** as part of the IBM Internship Program via Edunet.com.

We utilized IBM’s official documentation, the **watsonx.ai** platform, and the **Pixtral 12B** multimodal AI model, which provided essential AI capabilities and guidance.

---

Feel free to contribute, raise issues, or suggest improvements!

