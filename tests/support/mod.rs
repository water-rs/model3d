use serde_json::{Value, json};

pub fn document() -> Value {
    json!({
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0]}, {"nodes": [1]}],
        "nodes": [{"mesh": 0}, {"mesh": 0, "translation": [1000, 0, 0]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": 68}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 8},
            {"buffer": 0, "byteOffset": 44, "byteLength": 24}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3,
             "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3"}
        ],
        "animations": [{
            "name": "translate",
            "samplers": [{"input": 1, "output": 2, "interpolation": "LINEAR"}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]
        }]
    })
}

pub fn glb(document: &Value) -> Vec<u8> {
    let mut json = serde_json::to_vec(document).expect("serialize fixture");
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let data = [
        0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let binary: Vec<u8> = data.into_iter().flat_map(f32::to_le_bytes).collect();
    let total = u32::try_from(12 + 8 + json.len() + 8 + binary.len()).unwrap();
    let mut bytes = Vec::new();
    for word in [
        0x4654_6c67_u32,
        2,
        total,
        u32::try_from(json.len()).unwrap(),
        0x4e4f_534a,
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&u32::try_from(binary.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&0x004e_4942_u32.to_le_bytes());
    bytes.extend_from_slice(&binary);
    bytes
}
