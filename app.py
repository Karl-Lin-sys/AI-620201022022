"""
ProteinAI Web Dashboard
=======================
Flask-based web interface for protein analysis.
"""

from flask import Flask, render_template, request, jsonify
import sys
sys.path.insert(0, '.')

from src.protein_llm import ProteinLLM
from src.protein_gcn import ProteinStructureAnalyzer
from src.protein_diffusion import ProteinDiffusion
from src.protein_rl import ProteinRL
from src.protein_utils import analyze_sequence, validate_sequence

app = Flask(__name__)

# Initialize models (lazy loading)
_models = {}

def get_llm():
    if 'llm' not in _models:
        _models['llm'] = ProteinLLM()
    return _models['llm']

def get_gcn():
    if 'gcn' not in _models:
        _models['gcn'] = ProteinStructureAnalyzer()
    return _models['gcn']

def get_diffusion():
    if 'diffusion' not in _models:
        _models['diffusion'] = ProteinDiffusion()
    return _models['diffusion']

def get_rl():
    if 'rl' not in _models:
        _models['rl'] = ProteinRL()
    return _models['rl']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.json
    sequence = data.get('sequence', '').upper().strip()
    
    valid, msg = validate_sequence(sequence)
    if not valid:
        return jsonify({'error': msg}), 400
    
    # Basic properties
    props = analyze_sequence(sequence)
    
    # Function prediction
    llm = get_llm()
    functions = llm.predict_function(sequence, top_k=5)
    
    # Structure analysis
    gcn = get_gcn()
    structure = gcn.analyze_structure(sequence)
    
    return jsonify({
        'properties': props,
        'functions': [{'name': f, 'prob': p} for f, p in functions],
        'structure': [{'class': c, 'prob': p} for c, p in sorted(structure.items(), key=lambda x: -x[1])[:5]]
    })


@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    count = min(data.get('count', 3), 10)
    length = min(data.get('length', 25), 100)
    
    diffusion = get_diffusion()
    sequences = diffusion.sample(count, length)
    
    results = []
    for seq in sequences:
        props = analyze_sequence(seq)
        results.append({
            'sequence': seq,
            'length': len(seq),
            'hydropathy': props['hydropathy']
        })
    
    return jsonify({'sequences': results})


@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.json
    target_hydro = data.get('hydropathy', 0)
    target_charge = data.get('charge', 0)
    
    rl = get_rl()
    results = rl.optimize({'hydro': target_hydro, 'charge': target_charge}, steps=20)
    
    best = max(results, key=lambda x: x[1])
    props = rl.env.compute_properties(best[0])
    
    return jsonify({
        'sequence': best[0],
        'reward': best[1],
        'properties': props
    })


if __name__ == '__main__':
    print("Starting ProteinAI Web Dashboard...")
    print("Access at: http://localhost:5000")
    app.run(debug=True, port=5000)
