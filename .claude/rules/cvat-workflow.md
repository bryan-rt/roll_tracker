# CVAT Annotation Workflow

Applies to: `src/training_pipeline/cvat*`, `src/training_pipeline/export*`

## Setup
- Hosted at app.cvat.ai
- Organization: "Roll Tracker"
- Project: "BJJ Training Data" (skeleton label created manually in UI)
- Skeleton label name: "Skeleton" (capital S)

## SDK usage (cvat-sdk 2.62)
```python
from cvat_sdk.api_client.models import TaskWriteRequest, PatchedLabelRequest
task_spec = TaskWriteRequest(name="task_name")
task_spec.project_id = project_id
task = client.tasks.create(task_spec)
task.upload_data(resources=["video.mp4"], params={"image_quality": 95})
```

Use `getattr(result, 'results', result)` for paginated API results.
Use `params` dict (not keyword args) for `upload_data`.

## Annotation workflow
1. Upload video to CVAT project (via SDK or manual)
2. Open job in annotation editor
3. Select skeleton tool → Track mode (NOT Shape)
4. Place all 17 keypoints per person, use occluded flag for hidden joints
5. Advance ~10-30 frames, correct interpolated skeletons
6. Annotate ALL people on mat per frame
7. Save frequently

## Annotation best practices
- Use Track mode for interpolation between keyframes
- Mark joints as occluded (v=1) when hidden by another body part
- Mark joints as visible (v=2) when you can locate the joint even at an angle
- Annotate every 10th frame for variety, let CVAT interpolate between keyframes
- Skip frames rather than partially annotate — each frame either fully annotated or not at all

## Export workflow
1. Actions → Export task dataset → "Ultralytics YOLO Pose 1.0"
2. Actions → Export task dataset → "Ultralytics YOLO Oriented Bounding Boxes 1.0"
3. Run `tools/merge_cvat_exports.py` to combine bboxes + remap keypoints to COCO order

## Known issues
- CVAT XML annotation IMPORT fails (IndexError server-side) — manual annotation only
- Format names for import: "CVAT 1.1"; for export: "CVAT for video 1.1"

## Multi-user annotation
- Invite collaborators via Organization → Invite members
- Set role to "Worker" for annotators
- Assign specific jobs to workers (Worker role can only see assigned jobs)
- Split video into multiple jobs via segment_size for parallel annotation
- Workers must switch to Organization workspace to see assigned tasks
