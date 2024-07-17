# styles.py

# -*- coding: utf-8 -*-
import streamlit as st


def color_panel(color1=None, color2=None, color3=None, content=None):
    st.markdown(
        f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:20px;font-family:Candara;">{content}</p>',
        unsafe_allow_html=True,
    )


def load_global_styles():
    return """
    <style>
    # @import url('https://fonts.googleapis.com/css2?family=Ваш_Шрифт:wght@400;700&display=swap');

    body, h1, h2, h3, h4, h5, h6, p, div, span, a, li, ul, ol, input, button, select, textarea {
        font-family: 'Candara', Candara;
        /* Добавьте другие глобальные стили по вашему желанию */
    }
    </style>
    """


def custom_css():
    return """
    <style>
    .custom-title {
        font-family: 'Candara', Candara;
        color: #000000;
        font-size: 76px;
        text-align: center;
    }
    .custom-title_2 {
        font-family: 'Candara', Candara;
        color: #000000;
        font-size: 54px;
        text-align: center;
    }
    .custom-title_3 {
        font-family: 'Candara', Candara;
        color: #000000;
        font-size: 44px;
        text-align: center;
    }
    .custom-title_5 {
        font-family: 'Candara';
        color: #000000;
        font-size: 28px;
        text-align: center;
    }
    .custom-title_4 {
        font-family: 'Candara';
        color: #000000;
        font-size: 18px;
    }
    .custom-title_6 {
        font-family: 'Candara';
        color: #000000;
        font-size: 15px;
    }
     .radio-text {
        font-size: 26px !important;  
    }
    </style>
    """
