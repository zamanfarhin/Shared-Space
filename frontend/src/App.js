import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedName, setSelectedName] = useState('Satyajit Ray');

  const names = [
    { label: 'Satyajit Ray', value: 'satyajit_ray' },
    { label: 'Orhan Pamuk', value: 'orhan_pamuk' },
    { label: 'Hayao Miyazaki', value: 'hayao_miyazaki' },
    { label: 'Humayun Ahmed', value: 'humayun_ahmed' }
  ];  

  const handleGenerate = async () => {
    console.log(`Generating for: ${selectedName}`);
    // Replace this with your fetch call to backend later
    // Example: await fetch(`/generate?name=${selectedName}`)
  };

  return (
    <div className="App">
      <h1>Shared-Space</h1>
      <div className="generator-box">
        <p className="prompt-text">a space shared by</p>
        <select
          className="dropdown"
          value={selectedName}
          onChange={(e) => setSelectedName(e.target.value)}
        >
          {names.map(({ label, value }) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
        <button className="generate-button" onClick={handleGenerate}>
          Generate
        </button>
      </div>
    </div>
  );
}

export default App;


