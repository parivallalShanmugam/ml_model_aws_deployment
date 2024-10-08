## **Project Title: Sentiment Analysis on Women's Clothing E-Commerce Reviews**

### **Project Overview**

The project aims to analyze customer reviews from a women's clothing e-commerce platform to classify sentiments (positive or negative) and uncover common themes or topics. The model was deployed as a service using FastAPI, Docker, and AWS.

### **Data Source**

- **Dataset**: *Womens Clothing E-Commerce Reviews.csv*
- Contains columns such as `Review Text` and `Sentiment`. This dataset provides a rich source of feedback on customer experiences.

### **Key Steps in the Project**

1. **Data Preprocessing**
    - **Text Cleaning**: Removed unwanted characters, stopwords, and applied tokenization to prepare the text data.
    - **Vectorization**: Used `Tokenizer` from Keras to convert text into numerical form by converting the review text into sequences and padding them to ensure uniform input size.
2. **Text Classification Models**
    - **LSTM Model**:
        - Built an LSTM-based deep learning model to capture sequential dependencies in the text data.
        - Input: Tokenized sequences of the reviews.
        - Model layers: Embedding layer, LSTM layer with dropout, fully connected Dense layers.
        - Output: Sentiment classification (positive or negative).
    - **RoBERTa Pre-trained Model**:
        - Leveraged Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment` to perform sentiment classification using pre-trained transformer-based models.
        - Fine-tuned on the e-commerce review dataset for improved performance on this specific domain.
3. **Topic Modeling**
    - **BERTopic**: Used for extracting key topics from customer reviews. This technique helped identify frequent themes discussed, such as quality, size, shipping, and customer service.
    - Enabled the extraction of insights to improve business decisions based on customer feedback trends.
4. **API Development**
    - **FastAPI**: Created an API endpoint to serve the model for real-time predictions.
    - Endpoint allows users to submit a review and get back a sentiment prediction.
5. **Model Deployment**
    - **Docker**:
        - Built a Docker image that packages the entire application, including the trained model, API, and necessary libraries.
        - Followed best practices like minimizing image size using multi-stage builds and specifying dependencies in a `requirements.txt` file.
    - **AWS Deployment**:
        - Deployed the Docker container to AWS (EC2 or ECS) to provide a live service.
        - The model is now accessible via a public URL, enabling predictions from anywhere.
        
        [**AWS Deployment Steps**](https://www.notion.so/AWS-Deployment-Steps-11922e244a2f801cb920d1c7e34a2142?pvs=21)
        

### **Challenges**

- Handling imbalanced data in sentiment analysis.
- Optimizing the model to perform well with both long and short customer reviews.
- Ensuring the deployment is scalable and robust.

### **Future Improvements**

- Integrate real-time feedback loops to continuously retrain the model on fresh data.
- Improve topic modeling to incorporate more complex aspects of customer sentiment.
