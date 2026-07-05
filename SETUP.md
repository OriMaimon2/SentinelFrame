# Setup Instructions

## 1. Install Dependencies

```bash
pip install google-genai openai
```

## 2. Configure API Keys

### Step 1: Create config.py from template
```bash
cp config.example.py config.py
```

### Step 2: Add your API keys to config.py

**Get OpenAI API Key:**
- Visit: https://platform.openai.com/account/api-keys
- Copy your API key
- Edit `config.py` and set `OPENAI_API_KEY`

**Get Google API Key:**
- Visit: https://aistudio.google.com/app/apikey
- Copy your API key
- Edit `config.py` and set `GOOGLE_API_KEY`

### Step 3: Verify config.py is in .gitignore
The file `.gitignore` already includes `config.py`, so your keys will **NOT** be committed to GitHub.

```bash
grep "config.py" .gitignore
# Should output: config.py
```

## 3. Run Scripts

### Generate 1 sample image per label
```bash
python generate_sample_images.py
```

Output:
- Images saved in `sample_images/{label}/`
- Metadata in `sample_images/sample_metadata.json`

### Test with a single image
```bash
python test_single_image.py
```

Edit these settings in the script:
- `MODEL_PROVIDER`: "openai" or "google"
- `LABEL`: which label to test
- `PROMPT`: the image generation prompt

## 4. Push to GitHub (Safely)

Your API keys are safe because:

✅ `config.py` is in `.gitignore`  
✅ Only `config.example.py` is committed  
✅ GitHub will not see your real keys  

```bash
git add .
git commit -m "Add config template and generation scripts"
git push
```

## Troubleshooting

### ImportError: No module named 'google'
```bash
pip install google-genai
```

### ImportError: No module named 'openai'
```bash
pip install openai
```

### FileNotFoundError: config.py not found
Make sure you created config.py:
```bash
cp config.example.py config.py
```

### 404 error from Google API
- Verify your Google API key is correct
- Check that you have the right model name
- Ensure your Google account has access to Gemini API
