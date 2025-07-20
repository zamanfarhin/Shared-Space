import React, { useState } from 'react';
import './App.css';

function App() {
  const [name, setName] = useState('Satyajit Ray');
  const [style, setStyle] = useState(null);

  const fetchStyle = async () => {
    const res = await fetch(`http://localhost:5000/generate-style?name=${encodeURIComponent(name)}`);
    const data = await res.json();
    setStyle(data);
  };

  return (
    <div className="App">
      <h1>Shared-Space</h1>
      <input value={name} onChange={e => setName(e.target.value)} />
      <button onClick={fetchStyle}>Generate</button>

      {style && !style.error && (
        <div className="style-output">
          <h2>{name}</h2>
          <p><strong>Themes:</strong> {style.themes.join(', ')}</p>
          <p><strong>Palette:</strong></p>
          <div style={{ display: 'flex', gap: '10px' }}>
            {style.palette.map(color => (
              <div key={color} style={{ background: color, width: '40px', height: '40px', borderRadius: '4px' }} />
            ))}
          </div>
          <p><strong>Moodboard:</strong></p>
          <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
            {style.moodboard.map((url, idx) => (
              <img key={idx} src={url} alt="Moodboard" width={100} />
            ))}
          </div>
        </div>
      )}

      {style && style.error && <p>{style.error}</p>}
    </div>
  );
}

export default App;


