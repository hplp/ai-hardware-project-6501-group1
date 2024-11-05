[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Buol6fpg)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=16897609)
# AI_Hardware_Project

## Team Name: 
6501_project_group 1

## Team Members:
- Mohamed Tajudeen Sunaideen
- Longhao Tan

## Project Title:
Natural Language Processing (NLP) Models for sentiment analysis based on FPGA

## Project Description:
Our project aims to deploy lightweight NLP models for sentiment analysis on Pynq-Z1 hardware. Sentiment analysis requires processing text data to determine the emotion within sentences (e.g., positive, negative, neutral). It's typically handled by complex NLP models like BERT. However it requires too many computational resources to be deployed on resource-limited devices. To solve this problem, we will fine-tune lightweight models — DistilBERT, TinyBERT, and MobileBERT — using the Amazon Reviews Dataset. Then we’ll deploy these optimized models on Pynq-Z1 and evaluate their inference time and accuracy. Finally we will compare them to their performance on CPU/GPU.

## Key Objectives:
- Fine-tune lightweight models like DistilBERT, TinyBERT, and MobileBERT on the Amazon Reviews Dataset to enhance sentiment analysis performance.
- Deploy those optimized models on Pynq-Z1 to achieve efficent, low-cost sentiment analysis.
- Measure and compare inference time and accuracy on Pynq-Z1 against them on traditional hardware (CPU/GPU) to demonstrate FPGA's viability for NLP tasks.

## Technology Stack:
Pynq-Z1, Xilinx Vitis AI, Python.

## Expected Outcomes:
- We expect our optimized lightweight models to achieve high accuracy on the Pynq-Z1 as BERT model.
- We expect to achieve significant reductions in inference time on FPGA compared to traditional CPU/GPU deployments

## Timeline:
(Provide a rough timeline or milestones for the project)
