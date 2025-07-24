#!/usr/bin/env python3
"""
Test script to verify the number input functionality fix
"""

import streamlit as st
from typing import Optional

def test_number_input():
    st.title("Number Input Test - Fixed Version")
    
    # Test Optional int
    st.subheader("Optional Integer Field")
    
    # Simulate the app's logic
    key_base = "test_int"
    k_mode = f"{key_base}_mode"
    k_val = f"{key_base}"
    
    # Initialize session state only if not already set (simulating the fix)
    if k_mode not in st.session_state:
        st.session_state[k_mode] = "(None)"  # Default to None
    if k_val not in st.session_state:
        st.session_state[k_val] = 0  # Default value
    
    st.write(f"Session state before radio: {k_mode} = {st.session_state.get(k_mode, 'NOT SET')}")
    
    modes = ["(None)", "Number"]
    choice = st.radio("Select mode:", modes, key=k_mode)
    
    st.write(f"Session state after radio: {k_mode} = {st.session_state.get(k_mode, 'NOT SET')}")
    st.write(f"Radio choice: {choice}")
    
    if choice == "(None)":
        st.write("Value: None")
    else:
        val = st.number_input(
            "Enter integer:",
            value=st.session_state[k_val],
            step=1,
            key=k_val
        )
        st.write(f"Value: {val}")
        st.write(f"Session state: {k_val} = {st.session_state.get(k_val, 'NOT SET')}")
    
    # Test Optional float
    st.subheader("Optional Float Field")
    
    key_base_float = "test_float"
    k_mode_float = f"{key_base_float}_mode"
    k_val_float = f"{key_base_float}"
    
    # Initialize session state only if not already set (simulating the fix)
    if k_mode_float not in st.session_state:
        st.session_state[k_mode_float] = "(None)"  # Default to None
    if k_val_float not in st.session_state:
        st.session_state[k_val_float] = 0.0  # Default value
    
    choice_float = st.radio("Select mode:", modes, key=k_mode_float)
    
    if choice_float == "(None)":
        st.write("Value: None")
    else:
        val_float = st.number_input(
            "Enter float:",
            value=st.session_state[k_val_float],
            step=1.0,
            key=k_val_float
        )
        st.write(f"Value: {val_float}")
    
    # Show all session state for debugging
    st.subheader("Session State Debug")
    for key, value in st.session_state.items():
        if key.startswith("test_"):
            st.write(f"{key}: {value}")

if __name__ == "__main__":
    test_number_input()
