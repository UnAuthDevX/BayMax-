from flask import Flask, render_template, request, jsonify
import pickle
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained('./final_model')
tokenizer = GPT2Tokenizer.from_pretrained('./final_tokenizer')
tokenizer.pad_token = tokenizer.eos_token  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('user-input')

    input_tokens = tokenizer(user_input, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_tokens['input_ids'],
            max_length=300,    
            top_p=0.9,         
            do_sample=True     
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = f"Chatbot: {decoded_output}"
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
