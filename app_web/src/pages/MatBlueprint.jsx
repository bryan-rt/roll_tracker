import React, { useState, useEffect } from 'react';
import { Stage, Layer, Rect, Text, Line } from 'react-konva';

function MatBlueprint() {
  const [unit, setUnit] = useState('m');
  const conversionFactor = unit === 'ft' ? 3.28084 : 1.0;
  const GRID_SPACING = 0.5 * conversionFactor;
  const VALIDATION_GRID_SPACING = 1.0 * conversionFactor;
  const BUFFER = 1 * conversionFactor;

  const VIEWPORT_WIDTH = 800;
  const VIEWPORT_HEIGHT = 600;

  const [form, setForm] = useState({
    label: '',
    width: 1,
    height: 1,
  });

  const [sections, setSections] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (sections.length === 0) {
      setScale(1);
      setOffset({ x: 0, y: 0 });
      return;
    }

    const padding = BUFFER;
    const minX = Math.min(...sections.map(s => s.x)) - padding;
    const minY = Math.min(...sections.map(s => s.y)) - padding;
    const maxX = Math.max(...sections.map(s => s.x + s.width)) + padding;
    const maxY = Math.max(...sections.map(s => s.y + s.height)) + padding;

    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;

    const scaleX = VIEWPORT_WIDTH / contentWidth;
    const scaleY = VIEWPORT_HEIGHT / contentHeight;
    const newScale = Math.min(scaleX, scaleY);

    setScale(newScale);
    setOffset({ x: minX, y: minY });
  }, [sections, conversionFactor]);

  const drawGridLines = (spacing, color = "#ccc") => {
    const lines = [];
    const startX = offset.x - (offset.x % spacing);
    const endX = offset.x + VIEWPORT_WIDTH / scale;
    const startY = offset.y - (offset.y % spacing);
    const endY = offset.y + VIEWPORT_HEIGHT / scale;

    for (let x = startX; x <= endX; x += spacing) {
      lines.push(
        <Line
          key={`v-${spacing}-${x}`}
          points={[x, startY, x, endY]}
          stroke={color}
          strokeWidth={0.5 / scale}
        />
      );
    }

    for (let y = startY; y <= endY; y += spacing) {
      lines.push(
        <Line
          key={`h-${spacing}-${y}`}
          points={[startX, y, endX, y]}
          stroke={color}
          strokeWidth={0.5 / scale}
        />
      );
    }

    return lines;
  };

  const handleDelete = () => {
    if (selectedIndex !== null) {
      const updated = [...sections];
      updated.splice(selectedIndex, 1);
      setSections(updated);
      setSelectedIndex(null);
    }
  };

  const handleReset = () => {
    setSections([]);
    setSelectedIndex(null);
    setForm({ label: '', width: 1, height: 1 });
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(sections, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mat_blueprint.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleImport = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target.result);
        if (Array.isArray(imported)) {
          setSections(imported);
        } else {
          alert("Invalid file format.");
        }
      } catch {
        alert("Error reading JSON file.");
      }
    };
    reader.readAsText(file);
  };

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 48px)' }}>
      <div style={{ width: '300px', padding: '1rem', backgroundColor: '#f5f5f5' }}>
        <h2>Mat Section Properties</h2>

        <label>
          Units:
          <select value={unit} onChange={(e) => setUnit(e.target.value)}>
            <option value="m">Meters</option>
            <option value="ft">Feet</option>
          </select>
        </label>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            const newSection = {
              ...form,
              x: 50,
              y: 50,
              width: form.width * conversionFactor,
              height: form.height * conversionFactor,
            };
            setSections([...sections, newSection]);
          }}
        >
          <label>
            Label:
            <input
              type="text"
              value={form.label}
              onChange={(e) => setForm({ ...form, label: e.target.value })}
            />
          </label>
          <br />

          <label>
            Width ({unit}):
            <input
              type="number"
              value={form.width}
              onChange={(e) => setForm({ ...form, width: parseFloat(e.target.value) })}
            />
          </label>
          <br />

          <label>
            Height ({unit}):
            <input
              type="number"
              value={form.height}
              onChange={(e) => setForm({ ...form, height: parseFloat(e.target.value) })}
            />
          </label>
          <br />

          <button type="submit">Add Section</button>
        </form>

        <button style={{ marginTop: '1rem' }} onClick={handleDelete} disabled={selectedIndex === null}>
          Delete Selected
        </button>

        <button style={{ marginTop: '0.5rem' }} onClick={handleReset}>
          Reset Layout
        </button>

        <button style={{ marginTop: '0.5rem' }} onClick={handleExport}>
          Export to JSON
        </button>

        <input
          type="file"
          accept="application/json"
          style={{ marginTop: '0.5rem' }}
          onChange={handleImport}
        />
      </div>

      <div style={{ flexGrow: 1, backgroundColor: '#e0e0e0' }}>
        <Stage
          width={VIEWPORT_WIDTH}
          height={VIEWPORT_HEIGHT}
          scaleX={scale}
          scaleY={scale}
          offsetX={offset.x}
          offsetY={offset.y}
        >
          <Layer>{drawGridLines(VALIDATION_GRID_SPACING, "#aaa")}</Layer>
          <Layer>{drawGridLines(GRID_SPACING)}</Layer>
          <Layer>
            {sections.map((sec, index) => (
              <React.Fragment key={index}>
                <Rect
                  x={sec.x}
                  y={sec.y}
                  width={sec.width}
                  height={sec.height}
                  fill="lightblue"
                  stroke={index === selectedIndex ? 'red' : 'black'}
                  strokeWidth={1 / scale}
                  dash={index === selectedIndex ? [6 / scale, 3 / scale] : []}
                  draggable
                  onClick={() => setSelectedIndex(index)}
                  onDragEnd={(e) => {
                    const snap = GRID_SPACING;
                    const snappedX = Math.round(e.target.x() / snap) * snap;
                    const snappedY = Math.round(e.target.y() / snap) * snap;
                    const updated = [...sections];
                    updated[index] = { ...updated[index], x: snappedX, y: snappedY };
                    setSections(updated);
                  }}
                />
                <Text
                  text={sec.label}
                  x={sec.x + 5}
                  y={sec.y + 5}
                  fontSize={14 / scale}
                  fill="black"
                />
              </React.Fragment>
            ))}
          </Layer>
        </Stage>
      </div>
    </div>
  );
}

export default MatBlueprint;
