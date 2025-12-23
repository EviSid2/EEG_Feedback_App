🧠 EEG Insight App — Bachelor Thesis Study

Welcome to the EEG Insight App, an interactive tool designed for a user study examining personalized neurofeedback insights derived from EEG recordings.
This app is built with Python and Streamlit and can be deployed online so participants can use it without installing anything.

🚀 Features

--Start Page--
Upload or select EEG sessions

In study mode, users choose one of several predefined example recordings.

Alternatively, they can upload their own exported OpenBCI .csv files.

--Personalization form--

Participants enter demographic + contextual details to customize feedback.
Tthe only mandatory field is the task selection. Everything else is optional and only there for a more personalized result. Thus this step can be skipped.

--Multiple visualization modes--

Visual Feedback (orb/splatter/graph representation)

Literal Feedback (metaphor-poem/text-based interpretation)

LLM Coach (prewritten, study-mode AI feedback)

A short AI Coach feedback can be selected for Visual/Literal Mode. 

--Study Mode--

When enabled, all AI responses come from stored .html files

Participants never need to run an AI model locally

Ensures consistent feedback across the user study

--Deactivated Study Mode / Installed--

When the study mode is deactivated the user can still upload files manually or add the file in the Recordings folder directly. The newest file is then picked.
In personalization, the only mandatory field is the task selection. Everything else is optional and only there for a more personalized result (from the AI Coach). Thus this step can be skipped.
With the deactivated study mode -> realtime feedback from (downloaded) AI Coach (Ollama) is displayed
