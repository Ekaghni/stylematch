# first we import tkinter library and give it an alias tk for easy access
import tkinter as tk
# now import messagebox and scrolledtext from tkinter for dialogs and text areas
from tkinter import messagebox, scrolledtext
# import TfidfVectorizer from sklearn to convert text into numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# import cosine_similarity to measure how similar two text vectors are
from sklearn.metrics.pairwise import cosine_similarity

# let's define our first function calculate_similarity that takes two text parameters
def calculate_similarity(text1, text2):
    # add a docstring explaining this function calculates similarity using character n-grams
    """Calculate similarity using character n-grams (standard method for style detection)"""
    # check if either text is empty and return None if true
    if not text1 or not text2:
        # return None to indicate we cannot proceed
        return None
    
    # check if either text has less than 10 words by splitting and counting
    if len(text1.split()) < 10 or len(text2.split()) < 10:
        # show a warning messagebox telling user texts need at least 10 words
        messagebox.showwarning("Warning", "Texts should have at least 10 words for accurate analysis")
    
    # create a TfidfVectorizer object with character-level analysis
    vectorizer = TfidfVectorizer(
        # set analyzer to char to analyze character patterns instead of words
        analyzer='char',      # Use characters instead of words
        # set ngram_range to 2 comma 4 to capture sequences of 2, 3 and 4 characters
        ngram_range=(2, 4),   # Look at 2, 3, and 4 character patterns
        # set lowercase to True to normalize all text
        lowercase=True
    )
    
    # start a try block to handle any errors during calculation
    try:
        # fit the vectorizer on both texts and transform them into a TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # calculate cosine similarity between the two text vectors and extract the score
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # return the calculated similarity score
        return similarity
    # catch any exceptions that might occur
    except:
        # return zero point zero as a fallback
        return 0.0

# define interpret_similarity function that takes a score parameter
def interpret_similarity(score):
    # add docstring to explain this function interprets the numerical score
    """Interpret the similarity score"""
    # check if score is 70 percent or higher
    if score >= 0.70:
        # return message indicating very similar, likely same author
        return "Very Similar - Likely Same Author"
    # check if score is between 50 and 70 percent
    elif score >= 0.50:
        # return message indicating somewhat similar, possibly same author
        return "Somewhat Similar - Possibly Same Author or Related Topics"
    # if score is below 50 percent
    else:
        # return message indicating different, likely different authors
        return "Different - Likely Different Authors or Topics"

# define compare_texts function which is our main comparison function
def compare_texts():
    # add docstring explaining this is the main function to compare texts
    """Main function to compare the two text inputs"""
    # get all text from first text box starting from line 1 position 0 to end and strip whitespace
    text1 = text_box1.get("1.0", tk.END).strip()
    # get all text from second text box starting from line 1 position 0 to end and strip whitespace
    text2 = text_box2.get("1.0", tk.END).strip()
    
    # check if either text box is empty
    if not text1 or not text2:
        # show error dialog asking user to enter text in both boxes
        messagebox.showerror("Error", "Please enter text in both boxes")
        # exit the function early
        return
    
    # call calculate_similarity function with both texts and store the result
    similarity_score = calculate_similarity(text1, text2)
    
    # check if we got a valid similarity score back
    if similarity_score is not None:
        # get the interpretation of this score by calling interpret_similarity
        interpretation = interpret_similarity(similarity_score)
        
        # create result text starting with similarity score formatted to 4 decimal places
        result_text = f"Similarity Score: {similarity_score:.4f}\n\n"
        # add the interpretation to result text with double newline
        result_text += f"Interpretation: {interpretation}\n\n"
        # add word count for first text by splitting and counting words
        result_text += f"Word Count - Text 1: {len(text1.split())} words\n"
        # add word count for second text by splitting and counting words
        result_text += f"Word Count - Text 2: {len(text2.split())} words"
        
        # update the result label with our formatted result text
        result_label.config(text=result_text)

# define clear_all function to reset everything back to default
def clear_all():
    # add docstring explaining this function clears all inputs and results
    """Clear all text boxes and results"""
    # delete everything from first text box from position 1.0 to end
    text_box1.delete("1.0", tk.END)
    # delete everything from second text box from position 1.0 to end
    text_box2.delete("1.0", tk.END)
    # reset result label back to default message
    result_label.config(text="Results will appear here")

# create the main window by calling tk dot Tk
root = tk.Tk()
# set the window title to AI Writing Style Detector
root.title("AI Writing Style Detector")
# set window dimensions to 900 pixels wide by 600 pixels tall
root.geometry("900x600")
# configure window background color to light gray
root.configure(bg="#f0f0f0")

# create a title label for the top of our application
title_label = tk.Label(root, text="AI Writing Style Detector", 
                       # set font to Arial size 20 with bold style and light gray background
                       font=("Arial", 20, "bold"), bg="#f0f0f0")
# pack the title label with 10 pixels padding on top and bottom
title_label.pack(pady=10)

# create a frame to hold both text input boxes
text_frame = tk.Frame(root, bg="#f0f0f0")
# pack the text frame with 10 pixels vertical and 20 pixels horizontal padding
text_frame.pack(pady=10, padx=20)

# create label for first text box saying Text Sample 1
label1 = tk.Label(text_frame, text="Text Sample 1", font=("Arial", 12, "bold"), bg="#f0f0f0")
# place label1 at row 0 column 0 with 10 pixels horizontal padding
label1.grid(row=0, column=0, padx=10)

# create first scrolled text widget that's 40 characters wide and 15 lines tall
text_box1 = scrolledtext.ScrolledText(text_frame, width=40, height=15, 
                                       # set font to Arial size 10 with word wrapping enabled
                                       font=("Arial", 10), wrap=tk.WORD)
# place text_box1 at row 1 column 0 with 10 pixels horizontal padding
text_box1.grid(row=1, column=0, padx=10)

# create label for second text box saying Text Sample 2
label2 = tk.Label(text_frame, text="Text Sample 2", font=("Arial", 12, "bold"), bg="#f0f0f0")
# place label2 at row 0 column 1 with 10 pixels horizontal padding
label2.grid(row=0, column=1, padx=10)

# create second scrolled text widget that's 40 characters wide and 15 lines tall
text_box2 = scrolledtext.ScrolledText(text_frame, width=40, height=15, 
                                       # set font to Arial size 10 with word wrapping enabled
                                       font=("Arial", 10), wrap=tk.WORD)
# place text_box2 at row 1 column 1 with 10 pixels horizontal padding
text_box2.grid(row=1, column=1, padx=10)

# create a frame to hold our action buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
# pack the button frame with 10 pixels vertical padding
button_frame.pack(pady=10)

# create compare button with text Compare Texts
compare_button = tk.Button(button_frame, text="Compare Texts", 
                           # set command to compare_texts function with Arial size 12 bold font
                           command=compare_texts, font=("Arial", 12, "bold"),
                           # set green background white text and width of 15 characters
                           bg="#4CAF50", fg="white", width=15)
# place compare button at row 0 column 0 with 10 pixels horizontal padding
compare_button.grid(row=0, column=0, padx=10)

# create clear button with text Clear All
clear_button = tk.Button(button_frame, text="Clear All", 
                         # set command to clear_all function with Arial size 12 bold font
                         command=clear_all, font=("Arial", 12, "bold"),
                         # set red background white text and width of 15 characters
                         bg="#f44336", fg="white", width=15)
# place clear button at row 0 column 1 with 10 pixels horizontal padding
clear_button.grid(row=0, column=1, padx=10)

# create a frame for displaying results with white background and raised border
result_frame = tk.Frame(root, bg="white", relief=tk.RAISED, borderwidth=2)
# pack result frame with padding and make it expand to fill available space
result_frame.pack(pady=10, padx=20, fill=tk.BOTH)

# create result label with default message Results will appear here
result_label = tk.Label(result_frame, text="Results will appear here", 
                        # set font to Arial size 11 white background left-aligned text
                        font=("Arial", 11), bg="white", justify=tk.LEFT,
                        # add 20 pixels padding on all sides
                        padx=20, pady=20)
# pack the result label to display it
result_label.pack()

# create instructions string telling users how to use the application
instructions = "Instructions: Paste two text samples (at least 10 words each) and click 'Compare Texts'"
# create instruction label with the instructions text in Arial size 9
instruction_label = tk.Label(root, text=instructions, font=("Arial", 9), 
                            # set light gray background with darker gray text
                            bg="#f0f0f0", fg="#666")
# pack instruction label with 5 pixels vertical padding
instruction_label.pack(pady=5)

# start the tkinter main event loop to run the application
root.mainloop()