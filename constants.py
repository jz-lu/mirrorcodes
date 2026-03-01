"""
`constants.py`

Code file storing all the constants, e.g. filename conventions, 
used in this library. There are also some functions which generate constants.
"""

STAGEPFX = 'STAGE'

def generate_identifier(n):
    return f'n{n}'

def get_filename(stage, n, r = None, abelian = True):
    abelianPFX = '' if abelian else 'NA_'
    if (stage == 3 and r is not None) or (stage == 1 and r is not None and r >= 0):
        return f'{abelianPFX}{STAGEPFX}{stage}_{generate_identifier(n)}_part{r}.pkl'
    return f'{abelianPFX}{STAGEPFX}{stage}_{generate_identifier(n)}.pkl'

RATE_THRESHOLD = 1/30
DISTANCE_THRESHOLD = 4

DISTANCE_RATE_THRESHOLD = 0.3 # const c such that we want R*d > c*n
