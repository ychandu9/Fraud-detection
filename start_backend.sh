#!/bin/bash
cd backend
source venv/bin/activate 2>/dev/null || python -m venv venv && source venv/bin/activate
pip install flask flask-cors numpy pandas scikit-learn twilio --quiet
python app.py
