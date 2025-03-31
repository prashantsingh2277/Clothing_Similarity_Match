# Fashion Recommendation System

This project implements a fashion recommendation system that:
1. Removes backgrounds from input images
2. Detects and extracts clothing items
3. Generates feature vectors for clothing items
4. Finds similar items from a database using cosine similarity

## Project Structure

## Workflow Overview

1. `remove_bg.py`: Remove background from input image
2. `R_CNN.py`: Detect pose and extract clothing region
3. `Feat_xtract.py`: Extract feature vectors from clothing
4. `similarity_score.py`: Find similar items from database

## Detailed Component Explanation

# Fashion Recommendation System

## 📌 Table of Contents
1. [System Overview](#-system-overview)
2. [Architecture Diagram](#-architecture-diagram)
3. [Component Details](#-component-details)
   - [Background Removal](#1-background-removal)
   - [Pose Detection & Clothing Extraction](#2-pose-detection--clothing-extraction)
   - [Feature Extraction](#3-feature-extraction)
   - [Similarity Scoring](#4-similarity-scoring)
4. [Setup Guide](#-setup-guide)
5. [Usage Pipeline](#-usage-pipeline)
6. [Technical Specifications](#-technical-specifications)
7. [Limitations](#-limitations)

## 🌐 System Overview

A multi-stage pipeline that:
1. Removes backgrounds from fashion images
2. Detects human pose and extracts clothing regions
3. Generates deep feature representations
4. Finds visually similar items from a database

```mermaid
graph TD
    A[Input Image] --> B[Background Removal]
    B --> C[Pose Detection]
    C --> D[Clothing Extraction]
    D --> E[Feature Extraction]
    E --> F[Similarity Matching]
    F --> G[Top Recommendations]
