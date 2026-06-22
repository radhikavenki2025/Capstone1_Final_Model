import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quanqual

import plotly.express as px
import plotly.io as pio
import seaborn as sns

pio.renderers.default = "svg"


def Mentalhealth_ExamScore(st_habits1 ):

    avg_score = st_habits1.groupby('mental_health_rating')['exam_score'].mean().round(2)
    
    plt.figure(figsize=(8, 4))
    plt.plot(avg_score.index, avg_score.values, 
             color='steelblue', marker='o', linewidth=2, markersize=6)
    
    plt.title('Average Exam Score by Mental Health Rating')
    plt.xlabel('Mental Health Rating')
    plt.ylabel('Average Exam Score')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def Studyhrs_ExamScore(st_habits1 ):

    avg_score = st_habits1.groupby('study_hrs')['exam_score'].mean().round(2)
    
    plt.figure(figsize=(8, 4))
    plt.plot(avg_score.index, avg_score.values, 
             color='steelblue', marker='o', linewidth=2, markersize=6)
    
    plt.title('Average Exam Score by Study hrs')
    plt.xlabel('Study hrs')
    plt.ylabel('Average Exam Score')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def mediahrs_mentalhealth(st_habits1 ):

    avg_score = st_habits1.groupby('media_hrs')['mental_health_rating'].mean().round(2)
    
    plt.figure(figsize=(8, 4))
    plt.plot(avg_score.index, avg_score.values, 
             color='steelblue', marker='o', linewidth=2, markersize=6)
    
    plt.title('Mental Health Rating by Media hrs')
    plt.xlabel('Media hrs')
    plt.ylabel('Mental Health')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()    

def mentalhealth_Diet_Exercise(st_habits1):

    mental_health = st_habits1.groupby(['diet_quality', 'exercise_fq'])['mental_health_rating'].mean().reset_index()

    fig = px.line(mental_health, y="mental_health_rating", x="diet_quality",color='exercise_fq')

    fig.show()

def sleep_mentalhealth(st_habits1 ):

    avg_score = st_habits1.groupby('sleep_hrs')['mental_health_warning'].mean().round(2)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='mental_health_warning', y = 'sleep_hrs', data=st_habits1,hue='mental_health_warning', 
                palette='Set2' ,legend = False )
    

                
    plt.title(' Mental Health Rating by Sleep hrs')
    plt.xlabel('Mental Health Warning (0 = At Risk 1 = Okay)')
    plt.ylabel('Sleep Hrs')
    plt.tight_layout()
    plt.show()

def correlation_heatmap(st_habits1):
    plt.figure(figsize=(14, 10))
    corr = st_habits1.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', 
                center=0, linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    