"""Visibility of an unclipped projected body polygon, not a visible-mask estimate."""
import numpy as np


def projected_visibility(points, shape):
    """Both area and height fractions; neither is an authenticated clinical verdict.

    Original pixel centres have viewport edges at -0.5 and size-0.5.
    Full-body vertices may lie outside that viewport, including above the frame.
    """
    raw=np.asarray(points)
    if raw.dtype.kind not in 'fiu' or raw.shape!=(4,2):
        raise ValueError('Four numeric full-body corners required')
    p=raw.astype(float)
    if not np.isfinite(p).all() or len(np.unique(p,axis=0))!=4:
        raise ValueError('Finite distinct full-body corners required')
    if len(shape)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1 for v in shape):
        raise ValueError('Positive integer image dimensions required')
    with np.errstate(over='raise',invalid='raise'):
        try:
            edges=np.roll(p,-1,axis=0)-p
            following=np.roll(edges,-1,axis=0)
            cross=edges[:,0]*following[:,1]-edges[:,1]*following[:,0]
            if not ((cross>0).all() or (cross<0).all()):raise ValueError('Ordered convex full-body quadrilateral required')
            def area(poly):
                if len(poly)<3:return 0.
                poly=np.asarray(poly,float)
                return float(abs(np.dot(poly[:,0],np.roll(poly[:,1],-1))-np.dot(poly[:,1],np.roll(poly[:,0],-1)))/2)
            full=area(p)
            if full<=0 or not np.isfinite(full):raise ValueError('Degenerate full-body area')
            visible=p.tolist()
            for axis,bound,lower in ((0,-.5,True),(0,shape[1]-.5,False),(1,-.5,True),(1,shape[0]-.5,False)):
                clipped=[]
                def inside(point):return point[axis]>=bound if lower else point[axis]<=bound
                for i,q in enumerate(visible):
                    before=visible[i-1]
                    a,b=inside(before),inside(q)
                    if a!=b:
                        ratio=(bound-before[axis])/(q[axis]-before[axis])
                        intersection=(np.asarray(before)+ratio*(np.asarray(q)-before)).tolist()
                        intersection[axis]=bound;clipped.append(intersection)
                    if b:clipped.append(q)
                visible=clipped
            height=float(p[:,1].max()-p[:,1].min())
            visible_area=area(visible)
            vertical=(float(np.max(np.asarray(visible)[:,1])-np.min(np.asarray(visible)[:,1]))/height) if visible_area>0 else 0.
            fraction=float(np.clip(visible_area/full,0,1))
            return {'visible_projected_area_fraction':fraction,'visible_vertical_extent_fraction':float(np.clip(vertical,0,1)),
                    'full_projected_area_pixels_squared':full,'visible_polygon':visible,
                    'source_geometry':'unclipped_full_body_quadrilateral',
                    'clinical_verdict':'undetermined','clinical_validation':False}
        except FloatingPointError as exc:
            raise ValueError('Full-body geometry exceeds finite arithmetic') from exc
