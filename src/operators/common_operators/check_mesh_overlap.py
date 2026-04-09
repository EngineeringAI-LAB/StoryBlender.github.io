import bpy
from mathutils import Vector

def check_mesh_overlap(mesh_name_1: str, mesh_name_2: str) -> bool:
    """
    Determine whether two mesh objects overlap by testing if any vertex of one mesh
    lies inside the other's volume using multi-direction ray casting.

    The method casts rays from each vertex (in the target mesh's local space) along
    6 axis-aligned directions and counts hits. If the vertex registers hits in most
    directions (>= 5), it is considered inside. The test is run in both directions.

    Args:
        mesh_name_1: Name of the first mesh object in bpy.data.objects.
        mesh_name_2: Name of the second mesh object in bpy.data.objects.

    Returns:
        True if any vertex of either mesh is inside the other's volume. False otherwise.

    Raises:
        ValueError: If either object is missing, is not of type 'MESH',
                    or has no vertices.
        RuntimeError: If Blender is unable to switch to OBJECT mode for safe access.
    """
    obj1 = bpy.data.objects.get(mesh_name_1)
    obj2 = bpy.data.objects.get(mesh_name_2)

    missing = [n for n, o in ((mesh_name_1, obj1), (mesh_name_2, obj2)) if o is None]
    if missing:
        raise ValueError(f"Missing object(s): {', '.join(missing)}")

    if obj1.type != 'MESH' or obj2.type != 'MESH':
        raise ValueError("Both objects must be MESH types")

    if not obj1.data.vertices or not obj2.data.vertices:
        raise ValueError("One or both meshes have no vertices")

    # Ensure OBJECT mode for safe mesh data access
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as e:
        raise RuntimeError(f"Failed to switch to OBJECT mode: {e}") from e

    # Axis-aligned directions for robust containment heuristic
    directions = [
        Vector((1, 0, 0)), Vector((-1, 0, 0)),
        Vector((0, 1, 0)), Vector((0, -1, 0)),
        Vector((0, 0, 1)), Vector((0, 0, -1)),
    ]
    min_hits_inside = 5  # consider a vertex inside if it hits in most directions

    def any_vertex_inside(source_obj, target_obj) -> bool:
        mw_src = source_obj.matrix_world
        inv_mw_tgt = target_obj.matrix_world.inverted()

        # Iterate vertices and early-exit on first positive result
        for v in source_obj.data.vertices:
            v_world = mw_src @ v.co
            v_local = inv_mw_tgt @ v_world

            hit_count = 0
            for d in directions:
                result, _loc, _normal, _index = target_obj.ray_cast(v_local, d)
                if result:
                    hit_count += 1
                    if hit_count >= min_hits_inside:
                        return True
        return False

    # Check both directions
    if any_vertex_inside(obj1, obj2):
        return True
    if any_vertex_inside(obj2, obj1):
        return True

    return False