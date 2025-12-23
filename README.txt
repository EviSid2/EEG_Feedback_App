🧠 EEG Feedback App — What's The Verdict?

Welcome to the EEG Feedback App, designed for a user study examining neurofeedback presentation modalities of EEG recordings.
This app is built with Python and Streamlit and uses the localy installed Ollama model - qwen2.5:1.5b-instruct-q4_K_M - for the AI-Coach.

🚀 Features

--Start Page--
Upload or select an EEG recording session. The recordings need to be .CSV files.

The application automatically locates the newest file under OpenBCI_Exports/Recordings. 
Alternatively, you can use the manual upload option to select a recording. 

--Personalization form--

Enter demographic + contextual details to customize feedback.
The only mandatory field here is the task selection. Everything else is optional and only there for a more personalized feedback by the AI-Coach.
You need to consent to the anonymous use of your answers (even if you only entered the task) to continue.

--Multiple visualization modes--

Visual Feedback (calmness-orb + focus bar + graph representation)

Literal Feedback (metaphor/poem + technical explanations)

LLM Coach (real-time AI feedback)

You can select one of the three visualization modalities, but have the option to get back to this page and explore the rest.
An additional short AI-Coach feedback can be selected for Visual/Literal mode. 
There is an option to start over (go back to start) from any visualization mode.

--AI--

There is an option to refresh the AI feedback (only in the LLM Mode).
The Ollama model can be changed to a model of your choice, by simply adjusting the name. (Please note, that this only works smoothly with any other Ollama model. Other AI assistants may need more adjustments)
