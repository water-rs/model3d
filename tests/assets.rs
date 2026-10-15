//! Loads the hash-verified Khronos reference corpus fetched by the repository script.

use std::path::PathBuf;

use futures_util::{StreamExt, stream};
use waterui_model3d::Model;

#[test]
fn pinned_material_skin_and_morph_assets_load() {
    let directory = std::env::var_os("MODEL3D_TEST_ASSETS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models"),
        PathBuf::from,
    );
    let cases = [
        ("BoxTextured", 0),
        ("DamagedHelmet", 0),
        ("AlphaBlendModeTest", 0),
        ("Fox", 3),
        ("AnimatedMorphCube", 1),
    ];
    // At most four asset decodes run together on the blocking executor.
    // The caller supplies files verified by scripts/fetch_models.py.
    let results: Vec<_> = pollster::block_on(
        stream::iter(cases)
            .map(|(name, clips)| {
                let path = directory.join(format!("{name}.glb"));
                async move { (name, clips, Model::load(path).await) }
            })
            .buffer_unordered(4)
            .collect(),
    );
    assert_eq!(results.len(), cases.len());
    for (name, clips, result) in results {
        let model = result.unwrap_or_else(|error| {
            panic!("{name}: {error}; fetch corpus with scripts/fetch_models.py");
        });
        assert!(model.bounds().is_valid(), "{name} has invalid scene bounds");
        assert_eq!(model.clip_count(), clips, "{name} animation inventory");
    }
}
