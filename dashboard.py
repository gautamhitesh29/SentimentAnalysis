from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import speech_recognition as sr
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import *
import time


class Dashboard2:
    def __init__(self, window):
        self.content = ''
        self.sentiment = ['Positive', 'Neutral', 'Negative']
        self.rank = []
        self.window = window
        self.window.title("Sentiment Analysis")
        self.window.geometry("1366x730")
        self.window.resizable(0, 0)
        self.window.state('iconic')
        self.window.config(background='#eff5f6')
        # Window Icon Photo
        icon = PhotoImage(file='images\\pic-icon.png')
        self.window.iconphoto(True, icon)

        # ==============================================================================
        # ================== HEADER ====================================================
        # ==============================================================================
        self.header = Frame(self.window, bg='#009df4')
        self.header.place(x=300, y=0, width=1070, height=60)
        self.header_label = Label(self.header, text="Sentiment Analysis", bg='#009df4', font=("", 25, "bold"),
                                     fg='white')
        self.header_label.place(x=410, y=8)

        # ==============================================================================
        # ================== SIDEBAR ===================================================
        # ==============================================================================
        self.sidebar = Frame(self.window, bg='#ffffff')
        self.sidebar.place(x=0, y=0, width=300, height=750)

        # =============================================================================
        # ============= BODY ==========================================================
        # =============================================================================
        self.heading = Label(self.window, text='Results', font=("", 15, "bold"), fg='#0064d3', bg='#eff5f6')
        self.heading.place(x=325, y=70)

        # body frame 1
        self.bodyFrame1 = Frame(self.window, bg='#ffffff')
        self.bodyFrame1.place(x=328, y=110, width=1040, height=350)

        # body frame 2
        self.bodyFrame2 = Frame(self.window, bg='#7EE6DE')
        self.bodyFrame2.place(x=328, y=495, width=310, height=220)

        # body frame 3
        self.bodyFrame3 = Frame(self.window, bg='#53A5E2')
        self.bodyFrame3.place(x=680, y=495, width=310, height=220)

        # body frame 4
        self.bodyFrame4 = Frame(self.window, bg='#3A6AD0')
        self.bodyFrame4.place(x=1030, y=495, width=310, height=220)

        # ==============================================================================
        # ================== SIDEBAR ===================================================
        # ==============================================================================

        # logo
        self.logoImage = ImageTk.PhotoImage(file='images/hyy.png')
        self.logo = Label(self.sidebar, image=self.logoImage, bg='#ffffff')
        self.logo.place(x=70, y=80)

        # Name of brand/person
        self.brandName = Label(self.sidebar, text='Hitesh Sharma', bg='#ffffff', font=("", 15, "bold"))
        self.brandName.place(x=80, y=200)

        # Recording
        self.recordingImage = ImageTk.PhotoImage(file='images/recording.png')
        self.recording = Label(self.sidebar, image=self.recordingImage, bg='#ffffff')
        self.recording.place(x=35, y=289)

        self.recording_text = Button(self.sidebar, text="Record", bg='#ffffff', font=("", 13, "bold"), bd=0,
                                     cursor='hand2', activebackground='#ffffff',
                                     command=self.record_window)
        self.recording_text.place(x=80, y=287)

        # CSV_file
        self.csvImage = ImageTk.PhotoImage(file='images/export-csv.png')
        self.csvfile = Label(self.sidebar, image=self.csvImage, bg='#ffffff')
        self.csvfile.place(x=35, y=340)

        self.csv_text = Button(self.sidebar, text="Upload CSV File", bg='#ffffff', font=("", 13, "bold"), bd=0,
                               cursor='hand2', activebackground='#ffffff',
                               command=self.import_csv_data)
        self.csv_text.place(x=80, y=345)

        # Audio
        self.AudioImage = ImageTk.PhotoImage(file='images/icons_mp3.png')
        self.audio = Label(self.sidebar, image=self.AudioImage, bg='#ffffff')
        self.audio.place(x=35, y=402)

        self.audio_text = Button(self.sidebar, text="Upload Audio File", bg='#ffffff', font=("", 13, "bold"), bd=0,
                                 cursor='hand2', activebackground='#ffffff',
                                 command=self.import_audio_data)
        self.audio_text.place(x=80, y=402)

        # reset
        self.resetImage = ImageTk.PhotoImage(file='images/rotate-right.png')
        self.Reset = Label(self.sidebar, image=self.resetImage, bg='#ffffff')
        self.Reset.place(x=35, y=452)

        self.Reset_text = Button(self.sidebar, text="Reset", bg='#ffffff', font=("", 13, "bold"), bd=0,
                                 cursor='hand2', activebackground='#ffffff',
                                 command=self.reset_window)
        self.Reset_text.place(x=80, y=452)

        # Exit
        self.exitImage = ImageTk.PhotoImage(file='images/exit-sign-48.png')
        self.Exit = Label(self.sidebar, image=self.exitImage, bg='#ffffff')
        self.Exit.place(x=35, y=505)

        self.Exit_text = Button(self.sidebar, text="Exit", bg='#ffffff', font=("", 13, "bold"), bd=0,
                                cursor='hand2', activebackground='#ffffff',
                                command=self.window.destroy)
        self.Exit_text.place(x=80, y=505)

        # =============================================================================
        # ============= BODY ==========================================================
        # =============================================================================

        # Body Frame 1
        self.pieChart_label = Label(self.bodyFrame1, text="Pie Chart", bg='#ffffff', font=("", 25, "bold"))
        self.pieChart_label.place(x=690, y=70)

        # Graph
        self.graph_label = Label(self.bodyFrame1, text="Bar Graph", bg='#ffffff', font=("", 25, "bold"),)
        self.graph_label.place(x=40, y=70)

        # Body Frame 2
        self.total_word = Label(self.bodyFrame2, text='0', bg='#7EE6DE', font=("", 25, "bold"))
        self.total_word.place(x=120, y=100)

        self.total_wordImage = ImageTk.PhotoImage(file='images/mind-map-64.png')
        self.totalWord = Label(self.bodyFrame2, image=self.total_wordImage, bg='#7EE6DE')
        self.totalWord.place(x=220, y=0)

        self.totalWord_label = Label(self.bodyFrame2, text="Total Words", bg='#7EE6DE', font=("", 12, "bold"),
                                     fg='white')
        self.totalWord_label.place(x=5, y=5)

        # Body Frame 3
        self.Sentiment_left = Label(self.bodyFrame3, text='Sentiment', bg='#53A5E2', font=("", 25, "bold"))
        self.Sentiment_left.place(x=80, y=100)

        # left icon
        self.SentimentImage = ImageTk.PhotoImage(file='images/puzzled-64.png')
        self.Sentiment = Label(self.bodyFrame3, image=self.SentimentImage, bg='#53A5E2')
        self.Sentiment.place(x=220, y=0)

        self.Sentiment_label = Label(self.bodyFrame3, text="Sentiment", bg='#53A5E2', font=("", 12, "bold"),
                                     fg='white')
        self.Sentiment_label.place(x=5, y=5)

        # Body Frame 4
        self.statics = Label(self.bodyFrame4,
                             text="Positive Sentiment: %" +
                                  "\nNeutral Sentiment: %" +
                                  "\nNegative Sentiment: %",
                             bg='#3A6AD0', font=("", 15, "bold"))
        self.statics.place(x=40, y=80)

        self.statics_label = Label(self.bodyFrame4, text="Statics", bg='#3A6AD0', font=("", 12, "bold"),
                                   fg='white')
        self.statics_label.place(x=5, y=5)
        # Frame 4 icon
        self.staticsIcon_image = ImageTk.PhotoImage(file='images/combo-chart-50.png')
        self.staticsIcon = Label(self.bodyFrame4, image=self.staticsIcon_image, bg='#3A6AD0')
        self.staticsIcon.place(x=250, y=0)

        # date and Time
        self.clock_image = ImageTk.PhotoImage(file="images/time.png")
        self.date_time_image = Label(self.sidebar, image=self.clock_image, bg="white")
        self.date_time_image.place(x=88, y=20)

        self.date_time = Label(self.window)
        self.date_time.place(x=115, y=15)
        self.show_time()

    def show_time(self):
        self.time = time.strftime("%H:%M:%S")
        self.date = time.strftime('%d/%m/%Y')
        set_text = f"  {self.time} \n {self.date}"
        self.date_time.configure(text=set_text, font=("", 13, "bold"), bd=0, bg="white", fg="black")
        self.date_time.after(100, self.show_time)

    def record_audio(self):
        while True:
            self.content = ''
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(source)
                try:
                    self.content = r.recognize_google(audio)
                except Exception:
                    pass
                return self.analyze_text()

    def record_window(self):
        rec = Tk()
        rec.config(background='#ffffff')
        recording_button = Button(rec, text="Recording", bg='#ffffff', font=("", 13, "bold"), bd=0,
                                  cursor='hand2', activebackground='green',
                                  command=self.record_audio)
        recording_button.place(x=20, y=20)
        ok_button = Button(rec, text="CONTINUE", bg='#ffffff', font=("", 13, "bold"), bd=0,
                           cursor='hand2', activebackground='#ffffff',
                           command=rec.destroy)
        ok_button.place(x=20, y=60)
        rec.mainloop()

    def import_csv_data(self):
        csv_file_path = filedialog.askopenfilename(initialdir='test/',
                                                   filetypes=[("CSV file", '.csv')])
        dataset = pd.read_csv(csv_file_path)
        return self.analyze_csv(dataset)

    def import_audio_data(self):
        audio_file_path = filedialog.askopenfilename(initialdir='test/',
                                                     filetypes=[("WAV Audio file", '.wav'),
                                                                ("Mp3 file", '.mp3')])
        while True:
            self.content = ''
            r = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:  # open files
                audio = r.listen(source)
                try:
                    self.content = r.recognize_google(audio)
                except Exception:
                    pass
                return self.analyze_text()

    def analyze_text(self):
        countOfWords = len(self.content.split())
        self.total_word.configure(text=str(countOfWords))
        senti_analyzer = SentimentIntensityAnalyzer()
        senti_dict = senti_analyzer.polarity_scores(self.content)
        pos = senti_dict['pos'] * 100
        neu = senti_dict['neu'] * 100
        neg = senti_dict['neg'] * 100
        self.rank.append(round(pos, 2))
        self.rank.append(round(neu, 2))
        self.rank.append(round(neg, 2))
        self.pie_chart_plot()
        self.bar_plot()
        self.statics.configure(text="Positive Sentiment: " + str(round(pos, 2)) + "%" +
                                    "\nNeutral Sentiment: " + str(round(neu, 2)) + "%" +
                                    "\nNegative Sentiment: " + str(round(neg, 2)) + "%")
        compound = senti_dict['compound']
        if compound >= 0.05:
            self.Sentiment_left.configure(text="Positive")
            self.SentimentImage = ImageTk.PhotoImage(file='images/happy-64.png')
            self.Sentiment.configure(image=self.SentimentImage)
        elif compound <= -0.05:
            self.Sentiment_left.configure(text="Negative")
            self.SentimentImage = ImageTk.PhotoImage(file='images/sad-64.png')
            self.Sentiment.configure(image=self.SentimentImage)
        else:
            self.Sentiment_left.configure(text="Neutral")
            self.SentimentImage = ImageTk.PhotoImage(file='images/nerd-64.png')
            self.Sentiment.configure(image=self.SentimentImage)

    def analyze_csv(self, dataset):
        self.totalWord_label.configure(text="Total Rows")
        self.total_word.configure(text=str(dataset['review'].size))
        dataset = dataset.head(300)
        dataset.dropna()
        pos, neu, neg, compound = [], [], [], []
        for i in dataset.index:
            a = dataset['review'][i]
            senti_analyzer = SentimentIntensityAnalyzer()
            senti_dict = senti_analyzer.polarity_scores(dataset['review'][i])
            pos.append(senti_dict['pos'] * 100)
            neu.append(senti_dict['neu'] * 100)
            neg.append(senti_dict['neg'] * 100)
            compound.append(senti_dict['compound'])
        pos = sum(pos) / len(pos)
        neu = sum(neu) / len(neu)
        neg = sum(neg) / len(neg)
        self.rank.append(round(pos, 2))
        self.rank.append(round(neu, 2))
        self.rank.append(round(neg, 2))
        self.pie_chart_plot()
        self.bar_plot()
        compound = sum(compound) / len(compound)
        self.statics.configure(text="Positive Sentiment: " + str(round(pos, 2)) + "%" +
                                    "\nNeutral Sentiment: " + str(round(neu, 2)) + "%" +
                                    "\nNegative Sentiment: " + str(round(neg, 2)) + "%")
        compound = senti_dict['compound']
        if compound >= 0.05:
            self.Sentiment_left.configure(text="Positive")
            self.SentimentImage = ImageTk.PhotoImage(file='images/happy-64.png')
            self.Sentiment.configure(image=self.SentimentImage)
        elif compound <= -0.05:
            self.Sentiment_left.configure(text="Negative")
            self.SentimentImage = ImageTk.PhotoImage(file='images/sad-64.png')
            self.Sentiment.configure(image=self.SentimentImage)
        else:
            self.Sentiment_left.configure(text="Neutral")
            self.SentimentImage = ImageTk.PhotoImage(file='images/nerd-64.png')
            self.Sentiment.configure(image=self.SentimentImage)

    def pie_chart_plot(self):
        fig = plt.figure(figsize=(5, 5), dpi=100)
        fig.set_size_inches(5, 3.5)
        colors = ['#6050DC', '#D52DB7', '#FF2E7E', '#FF6B45', '#FFAB05']

        # Plot pie chart
        plt.pie(self.rank, explode=(0.1, 0.1, 0.1), labels=self.sentiment, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # creates the pie chart like a circle
        canvasbar = FigureCanvasTkAgg(fig, master=self.window)
        canvasbar.draw()
        canvasbar.get_tk_widget().place(x=1120, y=285, anchor=CENTER)  # show the pieChart on the output window
        pass

    def bar_plot(self):
        fig = plt.figure(figsize=(5, 3.5), dpi=100)
        labelpos = np.arange(len(self.sentiment))

        # This section formats the barchart for output
        plt.bar(labelpos, self.rank, align='center', alpha=1.0)
        plt.xticks(labelpos, self.sentiment)
        plt.ylabel('Percentage')
        plt.tight_layout(pad=2.2, w_pad=0.5, h_pad=0.1)
        plt.xticks(rotation=30, horizontalalignment="center")

        # Applies the values on the top of the bar chart
        for index, datapoints in enumerate(self.rank):
            plt.text(x=index, y=datapoints + 0.3, s=f"{datapoints}", fontdict=dict(fontsize=10), ha='center',
                     va='bottom')

        # This section draws a canvas to allow the barchart to appear in it
        canvasbar = FigureCanvasTkAgg(fig, master=self.window)
        canvasbar.draw()
        canvasbar.get_tk_widget().place(x=600, y=285, anchor=CENTER)  # show the barchart on the output window

    def reset_window(self):
        self.content = ''
        self.rank = []
        self.totalWord_label.configure(text="Total Words")
        self.total_word.configure(text="0")
        self.statics.configure(text="Positive Sentiment: %" +
                                    "\nNeutral Sentiment: %" +
                                    "\nNegative Sentiment: %")
        self.Sentiment_left.configure(text="Sentiment")
        self.SentimentImage = ImageTk.PhotoImage(file='images/puzzled-64.png')
        self.Sentiment.configure(image=self.SentimentImage)



def wind():
    window = Tk()
    Dashboard2(window)
    window.mainloop()


if __name__ == '__main__':
    wind()
