//! Generated glTF fixtures exercise the public asset and playback contracts.

use std::time::Duration;

use serde_json::{Value, json};
use waterui_model3d::{InvalidClipError, Model, Playback};

mod support;
use support::{document, glb};

fn load(document: &Value) -> Result<Model, waterui_model3d::LoadError> {
    pollster::block_on(Model::from_glb(glb(document)))
}

#[test]
fn selected_scene_bounds_exclude_other_scene_instances() {
    let model = load(&document()).expect("valid generated triangle");
    assert_eq!(model.bounds().min.to_array(), [0.0, 0.0, 0.0]);
    assert_eq!(model.bounds().max.to_array(), [1.0, 1.0, 0.0]);
}

#[test]
fn playback_rejects_a_clip_from_another_asset() {
    let first = load(&document()).unwrap();
    let second = load(&document()).unwrap();
    let handle = first.clip_handle(0).unwrap();
    let mut playback = Playback::new();
    assert!(!playback.is_playing());
    assert!(matches!(
        playback.select_clip(&second, handle.clone()),
        Err(InvalidClipError::WrongModel)
    ));
    playback.select_clip(&first, handle).unwrap();
    playback.seek(&first, Duration::from_millis(500)).unwrap();
    assert_eq!(playback.time(), Duration::from_millis(500));
}

#[test]
fn external_uri_and_unknown_required_extension_keep_their_diagnostics() {
    let mut external = document();
    external["buffers"][0]["uri"] = json!("sidecar.bin");
    assert!(matches!(
        load(&external),
        Err(waterui_model3d::LoadError::ExternalUri(_))
    ));
    let mut extension = document();
    extension["extensionsRequired"] = json!(["VENDOR_unknown_required"]);
    let error = load(&extension).expect_err("unknown required extension must fail");
    assert!(error.to_string().contains("VENDOR_unknown_required"));
}

#[test]
fn invalid_graph_and_animation_shape_fail_during_loading() {
    let mut cycle = document();
    cycle["nodes"][0]["children"] = json!([1]);
    cycle["nodes"][1]["children"] = json!([0]);
    assert!(load(&cycle).is_err(), "cyclic node graph was accepted");

    let mut short_output = document();
    short_output["accessors"][2]["count"] = json!(1);
    assert!(
        load(&short_output).is_err(),
        "mismatched animation keys were accepted"
    );
}
