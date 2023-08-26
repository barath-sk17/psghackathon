import os
from pathlib import Path
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from cryptography.fernet import Fernet  # For data encryption
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .forms import CreateForm
import seaborn as sns
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from .models import CreateFile
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import os
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
BASE_DIR = Path(__file__).resolve().parent.parent

# finding the job based on resume
def home(request):
    create_form=CreateForm()
    context={'form':create_form}
    return render(request,"jobmatching.html",context)

def createfile(request):
    if request.method=="POST":
        create_form=CreateForm(request.POST,request.FILES)
        if create_form.is_valid():
            print("\n\n\n Hello")
            #forms
            resume = request.FILES['resume_file']
            CreateFile.objects.create(resume=resume)
            
            pdf_files = [file for file in os.listdir(os.path.join(BASE_DIR, 'upload')) if file.endswith('.pdf')]
            if pdf_files:
                CV_File = open(os.path.join(BASE_DIR / 'upload',pdf_files[0]),"rb")
                Req_File = open(os.path.join(BASE_DIR / 'upload',"Requirement.pdf"),"rb")
                Script_CV = PyPDF2.PdfReader(CV_File)
                Script_Req = PyPDF2.PdfReader(Req_File)
                pages_CV = len(Script_CV.pages)
                pages_Req = len(Script_Req.pages)
                # Extract text from CV PDF
                Script_CV_text = []
                with pdfplumber.open(CV_File) as pdf:
                    for i in range(pages_CV):
                        page = pdf.pages[i]
                        text = page.extract_text()
                        print(text)
                        Script_CV_text.append(text)
                Script_CV_text = ''.join(Script_CV_text)
                CV_Clear = Script_CV_text.replace("\n", "")

                # Extract text from Requirement PDF
                Script_Req_text = []
                with pdfplumber.open(Req_File) as pdf:
                    for i in range(pages_Req):
                        page = pdf.pages[i]
                        text = page.extract_text()
                        print(text)
                        Script_Req_text.append(text)
                Script_Req_text = ''.join(Script_Req_text)
                Req_Clear = Script_Req_text.replace("\n", "")

                # Prepare text for similarity comparison
                Match_Test = [CV_Clear, Req_Clear]

                # Create CountVectorizer and cosine similarity
                cv = CountVectorizer()
                count_matrix = cv.fit_transform(Match_Test)
                cosine_sim = cosine_similarity(count_matrix)

                # Calculate and display match percentage
                MatchPercentage = cosine_sim[0][1] * 100
                MatchPercentage = round(MatchPercentage, 2)
                print('\n\n\nMatch Percentage is:', str(MatchPercentage) + '% to Requirement')
                a=str(MatchPercentage)
    return render(request,"sign.html",{'a':a})

def transition(request):
    
    # Read input data from CSV file
    
    input_file = open(os.path.join(BASE_DIR / 'upload',"transition.csv"),"r")
    df = pd.read_csv(input_file)
    
    # Convert role names to numerical values
    role_mapping = {role: index for index, role in enumerate(df['CurrentRole'].unique())}
    df['CurrentRole'] = df['CurrentRole'].map(role_mapping)


    # Separate features (current skills) and target (desired role)
    X = df.drop(['EmployeeID', 'CurrentRole'], axis=1)
    y = df['CurrentRole']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(random_state=20)
    clf.fit(X_train, y_train)

    # Predict the desired role for the employee (based on their current skills)
    employee_skills = df.iloc[0, 2:].values  # Replace with actual skill values
    predicted_role = clf.predict([employee_skills])[0]

    # Print the predicted role
    predicted_role_name = next(key for key, value in role_mapping.items() if value == predicted_role)
    print("Predicted Role:", predicted_role_name)

    # Create a dynamic dictionary to map roles to recommended skills
    recommended_skills = {}
    for role in df['CurrentRole'].unique():
        role_mask = df['CurrentRole'] == role
        skills_for_role = df.loc[role_mask, df.columns[2:]].columns[df.loc[role_mask, df.columns[2:]].iloc[0]].tolist()
        recommended_skills[role] = skills_for_role
    print(recommended_skills)

    a=[]
    # Sample tailored development plan based on the predicted role
    if predicted_role in recommended_skills.keys():
        print("Recommended Skills to Learn:")
        for skill in recommended_skills[predicted_role]:
            print(skill)
            a.append(skill)
    else:
        print("No specific skills recommended for this role.")
        
    return render(request,"roletransition.html",{'a':a,'b':predicted_role_name})
                    
def dynamiccand(request):
    resumeDataSet = pd.read_csv(r"C:\Users\Barath K\Downloads\UpdatedResumeDataSet.csv")
    resumeDataSet['cleaned_resume'] = ''
    resumeDataSet.head()
    print ("Displaying the distinct categories of resume and the number of records belonging to each category:\n\n")
    print (resumeDataSet['Category'].value_counts())
    plt.figure(figsize=(20,5))
    plt.xticks(rotation=90)
    ax=sns.countplot(x="Category", data=resumeDataSet)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
    plt.grid()

    def cleanResume(resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText
        
    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
    resumeDataSet.head()
    resumeDataSet_d=resumeDataSet.copy()
    

    oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
    totalWords =[]
    Sentences = resumeDataSet['Resume'].values
    cleanedSentences = ""
    for records in Sentences:
        cleanedText = cleanResume(records)
        cleanedSentences += cleanedText
        requiredWords = nltk.word_tokenize(cleanedText)
        for word in requiredWords:
            if word not in oneSetOfStopWords and word not in string.punctuation:
                totalWords.append(word)
        
    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(50)
    print(mostcommon)
    from sklearn.preprocessing import LabelEncoder

    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
    resumeDataSet_d.Category.value_counts()
    del resumeDataSet_d

    requiredText = resumeDataSet['cleaned_resume'].values
    requiredTarget = resumeDataSet['Category'].values

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english')
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)

    print ("Feature completed .....")

    X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,
                                                    shuffle=True, stratify=requiredTarget)
    x=[]
    y=[]
    print(X_train.shape)
    
    print(X_test.shape)
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
    print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
    
    return render(request,"candidate.html",{'a':a,'b':predicted_role_name})

def freelancer(request):
    # Generate synthetic data
    np.random.seed(42)

    num_freelancers = 1000
    num_projects = 500

    freelancer_data = {
        'Freelancer_ID': [1, 2, 3, 4, 5, 6],
        'Technical_Skills': ['Python, SQL', 'Java, Machine Learning', 'Python, Problem Solving','React', 'Django','Nodejs'],
        'Soft_Skills': ['Communication, Teamwork', 'Problem Solving', 'Communication, Teamwork','Project management', 'Tester', 'Developer']
    }
    freelancers = pd.DataFrame(freelancer_data)

    # Sample data for projects
    project_data = {
        'Project_ID': [101, 102, 103, 104, 105, 106],
        'Technical_Requirements': ['NodeJS', 'Machine Learning', 'Python, Problem Solving','React', 'Django','Java, Machine Learning'],
        'Soft_Skill_Requirements': ['Communication', 'Communication', 'Communication, Teamwork','Project management', 'Tester', 'Developer']
    }
    projects = pd.DataFrame(project_data)

    # Match freelancers with projects based on technical skills
    def match_freelancers_projects(freelancer_data, project_data):
        recommendations = {}
        for i in range(len(freelancer_data)):
            matching_projects = []
            freelancer_id = freelancer_data[i]['Freelancer_ID']
            freelancer_skills = set(freelancer_data[i]['Technical_Skills'])
            
            for project in project_data:
                project_id = project['Project_ID']
                project_requirements = set(project['Technical_Requirements'])
                
                if project_requirements.issubset(freelancer_skills):
                    matching_projects.append(project_id)
            
            recommendations[freelancer_id] = matching_projects
        return recommendations

    # Simplified encryption key generation (for demonstration purposes)
    encryption_key = Fernet.generate_key()

    # Encrypt sensitive data
    freelancers['Technical_Skills'] = freelancers['Technical_Skills'].apply(
        lambda x: x
    )
    projects['Technical_Requirements'] = projects['Technical_Requirements'].apply(
        lambda x: x
    )

    # Decrypt and match
    decrypted_freelancers = freelancers.copy()
    decrypted_freelancers['Technical_Skills'] = freelancers['Technical_Skills'].apply(
        lambda x: x
    )
    matching_results = match_freelancers_projects(decrypted_freelancers.to_dict('records'), projects.to_dict('records'))

    # Print matching results
    a=[]
    b=[]
    for freelancer_id, recommended_projects in matching_results.items():
        print(f"Freelancer {freelancer_id} should consider projects:", recommended_projects)
        a.append(freelancer_id)
        b.append(recommended_projects)  
    return render(request,"freelancer.html",{'a':a,'b':b})

def jobassign(request):
    # Generate synthetic data
    np.random.seed(42)

    num_freelancers = 1000
    num_projects = 500

    freelancer_data = {
        'Freelancer_ID': [1, 2, 3, 4, 5, 6],
        'Technical_Skills': ['Python, SQL', 'Java, Machine Learning', 'Python, Problem Solving','React', 'Django','Nodejs'],
        'Soft_Skills': ['Communication, Teamwork', 'Problem Solving', 'Communication, Teamwork','Project management', 'Tester', 'Developer']
    }
    freelancers = pd.DataFrame(freelancer_data)

    # Sample data for projects
    project_data = {
        'Project_ID': [101, 102, 103, 104, 105, 106],
        'Technical_Requirements': ['NodeJS', 'Machine Learning', 'Python, Problem Solving','React', 'Django','Java, Machine Learning'],
        'Soft_Skill_Requirements': ['Communication', 'Communication', 'Communication, Teamwork','Project management', 'Tester', 'Developer']
    }
    projects = pd.DataFrame(project_data)

    # Match freelancers with projects based on technical skills
    def match_freelancers_projects(freelancer_data, project_data):
        recommendations = {}
        for i in range(len(freelancer_data)):
            matching_projects = []
            freelancer_id = freelancer_data[i]['Freelancer_ID']
            freelancer_skills = set(freelancer_data[i]['Technical_Skills'])
            
            for project in project_data:
                project_id = project['Project_ID']
                project_requirements = set(project['Technical_Requirements'])
                
                if project_requirements.issubset(freelancer_skills):
                    matching_projects.append(project_id)
            
            recommendations[freelancer_id] = matching_projects
        return recommendations

    # Simplified encryption key generation (for demonstration purposes)
    encryption_key = Fernet.generate_key()

    # Encrypt sensitive data
    freelancers['Technical_Skills'] = freelancers['Technical_Skills'].apply(
        lambda x: x
    )
    projects['Technical_Requirements'] = projects['Technical_Requirements'].apply(
        lambda x: x
    )

    # Decrypt and match
    decrypted_freelancers = freelancers.copy()
    decrypted_freelancers['Technical_Skills'] = freelancers['Technical_Skills'].apply(
        lambda x: x
    )
    matching_results = match_freelancers_projects(decrypted_freelancers.to_dict('records'), projects.to_dict('records'))

    # Print matching results
    a=[]
    b=[]
    for freelancer_id, recommended_projects in matching_results.items():
        print(f"Freelancer {freelancer_id} should consider projects:", recommended_projects)
        a.append(freelancer_id)
        b.append(recommended_projects)  
    return render(request,"candidate.html",{'a':a,'b':b})
