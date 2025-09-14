# Create directories
New-Item -ItemType Directory -Path "src","research"

# Create files
New-Item -ItemType File -Path "src\__init__.py","src\helper.py","src\prompt.py",".env","setup.py","app.py","research\trials.ipynb","requirements.txt","README.md"

Write-Output "Directory and files structure created successfully."
