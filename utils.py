# Load the content of 'data/response.txt' into a string variable
def load_response_data(file_path):
    try:
        with open(file_path, 'r') as file:
            response_data = file.read()
        return response_data
    except Exception as e:
        return str(e)