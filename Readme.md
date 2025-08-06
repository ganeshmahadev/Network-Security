
-----

# Network Security Threat Detection using Machine Learning

This project implements an end-to-end machine learning pipeline to detect network security threats. The system ingests data, validates it, preprocesses it, and then trains, evaluates, and deploys a classification model. The entire pipeline is containerized using Docker and is designed for deployment on AWS.

## 🏛️ Architecture

The project follows a modular, multi-stage pipeline architecture. Data flows from a MongoDB database through several components, each responsible for a specific task. The final trained model is pushed to a cloud environment for serving. 

The main stages of the pipeline are:

1.  **Data Ingestion** 
2.  **Data Validation** 
3.  **Data Transformation** 
4.  **Model Training** 
5.  **Model Evaluation** 
6.  **Model Pushing & Deployment**

-----

## ✨ Features

  * **End-to-End ML Pipeline:** A complete workflow from data to a production-ready model.
  * **Data Validation:** Ensures data quality and integrity by checking for schema and data drift. 
  * **Automated Preprocessing:** Handles missing values and scales features for optimal model performance. 
  * **Model Factory:** Automatically selects the best-performing model based on predefined metrics. 
  * **Containerized:** Uses Docker for easy and consistent deployment across different environments. 
  * **CI/CD Ready:** Includes a GitHub Actions workflow for automated building and deployment. 
  * **Cloud Deployment:** Designed for deployment on AWS services like EC2 and ECR. 
-----

## 💻 Tech Stack

  * **Backend:** Python
  * **ML Libraries:** Scikit-learn, Pandas, NumPy
  * **Database:** MongoDB
  * **Containerization:** Docker
  * **Cloud Platform:** Amazon Web Services (AWS)
  * **CI/CD:** GitHub Actions

-----

## 🚀 Setup and Installation

### Prerequisites

  * Python 3.8+
  * Conda (or another virtual environment manager)
  * Docker
  * AWS CLI

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2\. Set Up the Environment

Create and activate a conda environment:

```bash
conda create --name network_security python=3.8 -y
conda activate network_security
```

### 3\. Install Dependencies

Install all the required Python packages:

```bash
pip install -r requirements.txt
```

### 4\. Environment Variables

Create a `.env` file in the root directory and add the following environment variables. These are essential for connecting to your database and AWS account.

```
MONGO_DB_URL="your_mongodb_connection_string"
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_REGION="your_aws_region"
```

-----

## 🏃 How to Run the Project

### Training the Pipeline

To run the entire machine learning pipeline, from data ingestion to model training, execute the main application script:

```bash
python main.py
```

This will generate all the necessary artifacts, including the trained model (`model.pkl`) and the data transformation object (`preprocessing.pkl`).

-----

## ⚙️ The Data Pipeline Explained

### 1. Data Ingestion 

  * **Process**: This stage connects to the MongoDB database to fetch the raw data.  It then exports the data to a feature store. Based on a schema file, unnecessary columns are dropped , and the data is split into training and testing sets. 
  * **Artifacts**: This component produces `train.csv` and `test.csv` files inside the data ingestion artifact directory.

### 2. Data Validation 

  * **Process**: This stage ensures the integrity of the data. It validates the number of columns , checks for missing columns, and verifies that numerical columns exist as expected.  A key part of this stage is detecting **Data Drift** , which compares the distribution of the current training data against a reference to ensure consistency. 
  * **Artifacts**: A status report is generated. If validation fails, it throws an error.

### 3. Data Transformation 

  * **Process**: The validated data is preprocessed to make it suitable for model training. This involves a `fit-transform` on the training data and a `transform` on the test data.  Key steps include:
      * **Handling Missing Values**: Using imputers like `KNNImputer`. 
      * **Feature Scaling**: Using `RobustScaler` to handle outliers.
      * **Handling Imbalanced Data**: Using techniques like `SMOTE`.
  * **Artifacts**: A transformation object (`preprocessing.pkl`) and the transformed train and test arrays are created. 

### 4. Model Training

  * **Process**: This stage takes the transformed data and trains the machine learning model. A **Model Factory** is used to find the best model based on expected accuracy. The best model is then trained on the full training dataset. 
  * **Artifacts**: The trained model is saved as `model.pkl`. 

-----

## ☁️ Deployment on AWS EC2

### Step 1: Build the Docker Image 

First, build the Docker image for the application. Make sure Docker is running on your local machine.

```bash
docker build -t network-security-app .
```

### Step 2: Push the Image to AWS ECR 

1.  **Create an ECR Repository**: Go to the AWS ECR console and create a new private repository named `network-security-app`.

2.  **Authenticate Docker to ECR**: Get the login command from the AWS CLI and execute it.

    ```bash
    aws ecr get-login-password --region your_aws_region | docker login --username AWS --password-stdin your_aws_account_id.dkr.ecr.your_aws_region.amazonaws.com
    ```

3.  **Tag the Docker Image**: Tag your local image with the ECR repository URI.

    ```bash
    docker tag network-security-app:latest your_aws_account_id.dkr.ecr.your_aws_region.amazonaws.com/network-security-app:latest
    ```

4.  **Push the Image**: Push the tagged image to your ECR repository.

    ```bash
    docker push your_aws_account_id.dkr.ecr.your_aws_region.amazonaws.com/network-security-app:latest
    ```

### Step 3: Launch an EC2 Instance

1.  Go to the AWS EC2 console and click "Launch instances".
2.  Choose an **Amazon Machine Image (AMI)**, such as Amazon Linux 2.
3.  Select an **instance type**, like `t2.micro` (eligible for the free tier).
4.  In the **Security Group** settings, add a rule to allow inbound HTTP traffic on port 80 (or the port your app uses).
5.  Launch the instance, making sure to create and save a new key pair for SSH access.

### Step 4: Deploy the App on the EC2 Instance

1.  **SSH into your EC2 instance**:

    ```bash
    ssh -i /path/to/your-key.pem ec2-user@your_ec2_public_ip
    ```

2.  **Install Docker on EC2**:

    ```bash
    sudo yum update -y
    sudo amazon-linux-extras install docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user
    ```

    You may need to log out and log back in for the group changes to take effect.

3.  **Log in to ECR from EC2**: Repeat the `aws ecr get-login-password` command from Step 2 on your EC2 instance. (You'll need to have AWS CLI configured on the instance, or use an IAM role with ECR permissions attached to the instance).

4.  **Pull the Docker Image from ECR**:

    ```bash
    docker pull your_aws_account_id.dkr.ecr.your_aws_region.amazonaws.com/network-security-app:latest
    ```

5.  **Run the Docker Container**:

    ```bash
    docker run -d -p 80:5000 your_aws_account_id.dkr.ecr.your_aws_region.amazonaws.com/network-security-app:latest
    ```

    This command runs the container in detached mode (`-d`) and maps port 80 on the EC2 instance to port 5000 in the container (assuming your Flask app runs on port 5000).

Your application should now be accessible via your EC2 instance's public IP address\!

-----

## 🔄 CI/CD with GitHub Actions 
This project is configured with a CI/CD pipeline using GitHub Actions. The workflow is defined in `.github/workflows/main.yml`. On every push to the `main` branch, the pipeline will automatically:

1.  Build the Docker image.
2.  Push the image to AWS ECR.
3.  (https://www.google.com/search?q=Optional) Trigger a deployment on a service like AWS App Runner or EC2.

-----

## 📂 Project Structure

```
.
├── .github/workflows/main.yml  # GitHub Actions CI/CD pipeline
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_trainer.py
│   │   └── model_pusher.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── training_pipeline.py
│   ├── utils/
│   │   └── main_utils.py
│   ├── config/
│   │   └── entity_config.py
│   ├── constants/
│   └── entity/
├── templates/
│   └── index.html
├── static/
├── .dockerignore
├── .env
├── .gitignore
├── Dockerfile
├── main.py
├── README.md
└── requirements.txt
```

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.