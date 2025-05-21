# Conversational Pydantic Generator

A streamlit application that uses multiple LLM agents to help generate Pydantic models through conversation.

## Features

- **Conversational Interface**: Describe your data model needs through a chat interface.
- **Multiple Specialized Agents**:
  - **Conversation Agent**: Gathers information about your data modeling needs
  - **Schema Generator**: Creates structured Pydantic classes and fields
  - **Advisor Agent**: Provides recommendations to improve the schema
  - **Structure Optimization Agent**: Suggests improved organization of your schema
  - **Verification Agent**: Ensures the schema meets best practices
- **Cost Tracking**: Monitor API usage and costs across all agents
- **Customizable LLM Settings**: Support for OpenAI, Azure OpenAI, and other compatible endpoints
- **Save & Restore**: Backup your conversation and schema for later use
- **Pricing Management**: Adjust pricing settings for different models

## Installation Guide

### Prerequisites

- Python 3.9 or higher
- An API key for OpenAI, Azure OpenAI, or a compatible service

### Installation Steps

#### For macOS/Linux:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/awesome_extraction_with_LLM_ExtraCTOps.git
   cd awesome_extraction_with_LLM_ExtraCTOps/apps/conversational_pydantic_generator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

#### For Windows:

1. Clone the repository:
   ```powershell
   git clone https://github.com/yourusername/awesome_extraction_with_LLM_ExtraCTOps.git
   cd awesome_extraction_with_LLM_ExtraCTOps\apps\conversational_pydantic_generator
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```

#### For Minerva Server (assuming Linux-based):

1. Connect to the server via SSH:
   ```bash
   ssh your-username@minerva-server-address
   ```

2. Navigate to your workspace and clone the repository:
   ```bash
   cd your-workspace
   git clone https://github.com/yourusername/awesome_extraction_with_LLM_ExtraCTOps.git
   cd awesome_extraction_with_LLM_ExtraCTOps/apps/conversational_pydantic_generator
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Make sure your virtual environment is activated
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and go to `http://localhost:8501`

## Using the Application

### Getting Started

1. **API Configuration**:
   - In the sidebar, select your LLM provider (OpenAI, Azure OpenAI, or OpenAI-Compatible)
   - Enter your API key and other required settings
   - Click "Check API Connection" to verify your setup

2. **Start the Conversation**:
   - In the main chat area, describe your data model needs
   - The conversation agent will ask questions to understand your requirements
   - Continue the conversation until you've provided enough information

3. **Generate and Refine the Schema**:
   - After enough information is collected, a schema will be automatically generated
   - Review the generated schema in the center column
   - Use the "Regenerate Schema" button if you'd like a fresh schema based on the conversation
   - Use the "Improve Structure" button to optimize the organization of your schema

4. **Get Recommendations**:
   - The advisor agent will automatically analyze your schema and provide recommendations
   - These are shown in the right column, grouped by priority
   - Consider these recommendations and continue the conversation to provide more details

5. **Finalize Your Schema**:
   - Once you're satisfied with the schema, click the "I'm Satisfied" button
   - The verification agent will perform a final check
   - If approved, you can download the Pydantic code

### Cost Management

The application tracks the cost of API calls:

1. View the total cost and breakdown in the sidebar
2. Adjust pricing settings for different models in the "Pricing Settings" section
3. Reset the cost counter when needed
4. Export a cost report for budgeting or reimbursement

### Saving and Restoring

1. **Save Your Session**:
   - Click "Save Conversation" in the sidebar
   - This will download a file with your entire session state

2. **Restore a Session**:
   - Upload a previously saved file using the "Restore from backup" uploader
   - Click "Refresh UI" after restoration

## Tips for Better Results

1. **Be Specific**: Provide as many details as possible about your data model
2. **Mention Relationships**: Clearly describe how different entities relate to each other
3. **Explain Validation Rules**: Mention any validation requirements for fields
4. **Iterate**: Use the conversation to refine your schema incrementally
5. **Try Structure Optimization**: If your schema becomes complex, use the structure optimization agent to suggest better organization

## Troubleshooting

- **API Connection Issues**: Verify your API key and connection settings
- **Schema Generation Errors**: Provide more specific information in the conversation
- **Slow Response**: Try using a smaller model for faster responses (e.g., gpt-4.1-mini instead of gpt-4o)
- **High Costs**: Monitor costs in the sidebar and adjust model selection as needed

## Privacy Considerations

- All conversations are processed through the selected LLM provider
- Your API key and conversations are not stored on our servers
- Consider sensitive information when using third-party LLM services

## Support and Feedback

If you encounter issues or have suggestions for improvement, please:
- Open an issue on the GitHub repository
- Contact the development team at [your-email@example.com]

## License

This project is licensed under the MIT License - see the LICENSE file for details.