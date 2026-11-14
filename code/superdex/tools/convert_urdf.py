"""Convert the PenSpin Allegro URDF for the SuperDex runtime loader, which ignores primitive collision shapes.

- every <box> collision becomes a closed 12-triangle .obj with the same size; the <origin> is kept on the element;
- a link with several collisions keeps the first and moves the others to new massless-free child links on fixed joints
  (no overlapping shells merged into one SDF);
- links without <inertial> but with a box collision get mass = volume * 1000 kg/m^3 (Isaac Gym default density) and
  the solid-box inertia;
- relative mesh paths are made absolute. Kinematics, limits and existing inertials are copied unchanged.
Writes <out_dir>/allegro_superdex.urdf, <out_dir>/meshes/*.obj and <out_dir>/conversion_report.json.
"""
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DENSITY = 1000.0


def box_obj(size):
    hx, hy, hz = (s / 2 for s in size)
    v = [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    # vertex index = 4*ix + 2*iy + iz (1-based in OBJ); outward-facing triangles
    f = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6), (1, 5, 6), (1, 6, 2),
         (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4)]
    return ''.join('v %.9g %.9g %.9g\n' % p for p in v) + ''.join('f %d %d %d\n' % t for t in f)


def main(src, out_dir):
    src, out = Path(src).resolve(), Path(out_dir).resolve()
    (out / 'meshes').mkdir(parents=True, exist_ok=True)
    root = ET.parse(src).getroot()
    report = {'source': str(src), 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
              'boxes_converted': 0, 'split_links': [], 'mass_added': {}}
    new_elems = []
    for link in root.findall('link'):
        name = link.get('name')
        cols = link.findall('collision')
        for k, col in enumerate(cols):
            geom = col.find('geometry')
            g = geom[0]
            if g.tag == 'box':
                size = [float(x) for x in g.get('size').split()]
                # no dots in the stem: the SuperDex loader silently drops mesh files like box_0.003_....obj
                mesh = out / 'meshes' / ('box_%s.obj' % '_'.join('%dum' % round(s * 1e6) for s in size))
                mesh.write_text(box_obj(size))
                geom.remove(g)
                ET.SubElement(geom, 'mesh', filename=str(mesh))
                report['boxes_converted'] += 1
            elif g.tag == 'mesh':
                p = Path(g.get('filename'))
                g.set('filename', str(p if p.is_absolute() else (src.parent / p).resolve()))
            else:
                raise ValueError((name, g.tag))
        if link.find('inertial') is None and cols:
            size = [float(x[:-2]) * 1e-6 for x in link.find('collision/geometry/mesh').get('filename').split('box_')[1][:-4].split('_')]
            m = DENSITY * size[0] * size[1] * size[2]
            o = cols[0].find('origin')
            inert = ET.SubElement(link, 'inertial')
            ET.SubElement(inert, 'origin', xyz=o.get('xyz', '0 0 0') if o is not None else '0 0 0', rpy='0 0 0')
            ET.SubElement(inert, 'mass', value='%.9g' % m)
            a, b, c = size
            ET.SubElement(inert, 'inertia', ixx='%.9g' % (m * (b * b + c * c) / 12), iyy='%.9g' % (m * (a * a + c * c) / 12),
                          izz='%.9g' % (m * (a * a + b * b) / 12), ixy='0', ixz='0', iyz='0')
            report['mass_added'][name] = m
        for k, col in enumerate(cols[1:], 1):
            link.remove(col)
            child = ET.Element('link', name=f'{name}_col{k}')
            c2 = copy.deepcopy(col)
            child.append(c2)
            joint = ET.Element('joint', name=f'{name}_col{k}_joint', type='fixed')
            ET.SubElement(joint, 'parent', link=name)
            ET.SubElement(joint, 'child', link=f'{name}_col{k}')
            ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
            new_elems += [child, joint]
            report['split_links'].append(f'{name}_col{k}')
    for v in root.iter('visual'):
        for g in v.iter('mesh'):
            p = Path(g.get('filename'))
            g.set('filename', str(p if p.is_absolute() else (src.parent / p).resolve()))
    for e in new_elems:
        root.append(e)
    dst = out / 'allegro_superdex.urdf'
    ET.ElementTree(root).write(dst)
    report['output_sha256'] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in report.items()}))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
