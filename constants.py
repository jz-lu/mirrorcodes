"""
`constants.py`

Code file storing all the constants, e.g. filename conventions, 
used in this library. There are also some functions which generate constants.
"""

STAGEPFX = 'STAGE'

def generate_identifier(n):
    return f'n{n}'

def get_filename(stage, n, r = None):
    if stage == 3 and r is not None:
        return f'{STAGEPFX}{stage}_{generate_identifier(n)}_part{r}.pkl'
    return f'{STAGEPFX}{stage}_{generate_identifier(n)}.pkl'

RATE_THRESHOLD = 1/20
DISTANCE_THRESHOLD = 4

DISTANCE_RATE_THRESHOLD = 0.4 # const c such that we want R*d > c*n
