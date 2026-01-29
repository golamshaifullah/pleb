"""
Legacy dataset-fix utilities extracted from FixDataset.ipynb.

These functions implement the original notebook's dataset correction features:
- TIM fixes: whitespace/padd cleanup, missing flag insertion (-be/-pta/-group/-sys), NUPPI splitting, overlap removal, missing tim INCLUDE updates.
- PAR fixes: ensure ephem/clk/ne_sw, add missing JUMPs, coordinate conversion helpers, optional param additions.

They are intentionally kept close to the notebook logic for parity.
"""

from __future__ import annotations

import os
import re
import shutil
from glob import glob
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# MeerKAT helper definition from the notebook
meerkat = {}
meerkat['recv'] = {'LBAND'}
meerkat['nchan'] = 16
meerkat['bw'] = 48
meerkat['band_top'] = 1650

# jumps_per_system mapping extracted from the notebook
jumps_per_system={'EFF.EBPP.1360.tim': ['EFF.EBPP.1360'],
 'EFF.EBPP.1410.tim': ['EFF.EBPP.1410'],
 'EFF.EBPP.2639.tim': ['EFF.EBPP.2639'],
 'EFF.P200.1380.tim': ['EFF.P200.1365', 'EFF.P200.1425'],
 'EFF.P217.1380.tim': ['EFF.P217.1365', 'EFF.P217.1425'],
 'EFF.S110.2487.tim': ['EFF.S110.2487'],
 'EFF.S60.4857.tim': ['EFF.S60.4857'],
 'GM_GWB_1460_100_b1_pre36.tim': ['GM_GWB_1460_100_b1.1372', 'GM_GWB_1460_100_b1.1397', 'GM_GWB_1460_100_b1.1422', 'GM_GWB_1460_100_b1.1447'],
 'GM_GWB_1460_200_b0_post36.tim': ['GM_GWB_1460_200_b0.1285', 'GM_GWB_1460_200_b0.1335', 'GM_GWB_1460_200_b0.1385', 'GM_GWB_1460_200_b0.1435'],
 'GM_GWB_500_100_b1_pre36.tim': ['GM_GWB_500_100_b1.406', 'GM_GWB_500_100_b1.418', 'GM_GWB_500_100_b1.431', 'GM_GWB_500_100_b1.443', 'GM_GWB_500_100_b1.456', 'GM_GWB_500_100_b1.468', 'GM_GWB_500_100_b1.481', 'GM_GWB_500_100_b1.493'],
 'GM_GWB_500_200_b1_post36.tim': ['GM_GWB_500_200_b1.312', 'GM_GWB_500_200_b1.337', 'GM_GWB_500_200_b1.362', 'GM_GWB_500_200_b1.387', 'GM_GWB_500_200_b1.412', 'GM_GWB_500_200_b1.437', 'GM_GWB_500_200_b1.462', 'GM_GWB_500_200_b1.487'],
 'JBO.DFB.1400.tim': ['JBO.DFB.1400'],
 'JBO.DFB.1520.tim': ['JBO.DFB.1520'],
 'JBO.DFB.5000.tim': ['JBO.DFB.5000'],
 'JBO.MK2.1520.2.tim': ['JBO.MK2.1520'],
 'JBO.MK2.1520.tim': ['JBO.MK2.1520.2'],
 'JBO.ROACH.1520.tim': ['JBO.ROACH.1420', 'JBO.ROACH.1620'],
 'LEAP.1396.tim': ['LEAP.1396'],
 'NRT.BON.1400.tim': ['NRT.BON.1400'],
 'NRT.BON.1600.tim': ['NRT.BON.1600'],
 'NRT.BON.2000.tim': ['NRT.BON.2000'],
 'NRT.NUPPI.1484.tim': ['NRT.NUPPI.1292', 'NRT.NUPPI.1420', 'NRT.NUPPI.1548', 'NRT.NUPPI.1676'],
 'NRT.NUPPI.1854.tim': ['NRT.NUPPI.1662', 'NRT.NUPPI.1790', 'NRT.NUPPI.1918', 'NRT.NUPPI.2046'],
 'NRT.NUPPI.2539.tim': ['NRT.NUPPI.2411', 'NRT.NUPPI.2667'],
 'WSRT.P1.1380.1.tim': ['WSRT.P1.1380.1'],
 'WSRT.P1.1380.2.tim': ['WSRT.P1.1380.2'],
 'WSRT.P1.2273.tim': ['WSRT.P1.2273.C'],
 'WSRT.P1.323.tim': ['WSRT.P1.323.C'],
 'WSRT.P1.367.tim': ['WSRT.P1.367.C'],
 'WSRT.P1.840.tim': ['WSRT.P1.840'],
 'WSRT.P2.1380.tim': ['WSRT.P2.1380'],
 'WSRT.P2.2273.tim': ['WSRT.P2.2273'],
 'WSRT.P2.350.tim': ['WSRT.P2.350']}

# backend_bw mapping extracted from the notebook
backend_bw = {'EFF.EBPP.1360.tim': 128.,
            'EFF.EBPP.1410.tim': 128.,
            'EFF.EBPP.2639.tim': 128.,
            'EFF.P200.1380.tim': 400.,
            'EFF.P217.1380.tim': 400.,
            'EFF.S110.2487.tim': 400.,
            'JBO.DFB.1400.tim': 128.,
            'JBO.DFB.1520.tim': 128.,
            'JBO.ROACH.1520.tim': 512.,
            'LEAP.1396.tim': 128.,
            'NRT.BON.1400.tim': 128.,
            'NRT.BON.1600.tim': 128.,
            'NRT.NUPPI.1484.tim': 512.,
            'NRT.NUPPI.1854.tim': 512.,
            'SRT.DFB.1400': 128.,
            'SRT.DFB.330': 64.,
            'SRT.ROACH.1400': 256.,
            'SRT.ROACH.330': 128.,
            'WSRT.P1.1380.2.tim': 64.,
            'WSRT.P1.1380.C.tim': 64.,
            'WSRT.P1.1380.tim': 64.,
            'WSRT.P1.2273.2.tim': 64.,
            'WSRT.P1.2273.C.tim': 64.,
            'WSRT.P1.2273.tim': 64.,
            'WSRT.P1.323.tim': 32.,
            'WSRT.P1.328.tim': 32.,
            'WSRT.P1.367.tim': 32.,
            'WSRT.P1.382.tim': 32.,
            'WSRT.P2.1380.tim': 128.,
            'WSRT.P2.2273.tim': 128.,
            'WSRT.P2.350.tim': 64.,
            'MK.LBAND.1420.tim': 778.}

# overlapped_timfiles mapping extracted from the notebook
overlapped_timfiles = {'EFF.P200.1380.tim': ['EFF.EBPP.1360.tim', 'EFF.EBPP.1410.tim'],
                       'EFF.P217.1380.tim': ['EFF.EBPP.1360.tim', 'EFF.EBPP.1410.tim'],
                       'EFF.S110.2487.tim': ['EFF.EBPP.2639.tim', 'EFF.EBPP.2639.tim'],
                       'JBO.ROACH.1520.tim': ['JBO.DFB.1400.tim', 'JBO.DFB.1520.tim'],
                       'JBO.DFB.1520.tim': ['JBO.DFB.1400.tim'],
                       'NRT.NUPPI.1484.tim': ['NRT.BON.1400.tim', 'NRT.BON.1600.tim'],
                       'WSRT.P2.350.tim': ['WSRT.P1.323.tim', 'WSRT.P1.328.tim', 'WSRT.P1.367.tim', 'WSRT.P1.382.tim'],
                       'WSRT.P2.1380.tim': ['WSRT.P1.1380.tim', 'WSRT.P1.1380.C.tim', 'WSRT.P1.1380.2.tim'],
                       'WSRT.P2.2273.tim': ['WSRT.P1.2273.tim', 'WSRT.P1.2273.C.tim', 'WSRT.P1.2273.2.tim'],
                       'LEAP.1396.tim': ['EFF.P200.1380.tim', 'EFF.P217.1380.tim', 'EFF.S110.2487.tim',
                                         'JBO.ROACH.1520.tim',
                                         'NRT.NUPPI.1484.tim', 'NRT.NUPPI.1854.tim',
                                         'SRT.DFB.1400', 'SRT.ROACH.1400',
                                         'WSRT.P2.1380.tim']}

# pta_systems mapping extracted from the notebook
pta_systems = {'EFF.EBPP.1360.tim': 'EPTA',
		'EFF.EBPP.1410.tim': 'EPTA',
		'EFF.EBPP.2639.tim': 'EPTA',
		'EFF.P200.1380.tim': 'EPTA',
		'EFF.P217.1380.tim': 'EPTA',
		'EFF.S110.2487.tim': 'EPTA',
		'JBO.DFB.1400.tim': 'EPTA',
		'JBO.DFB.1520.tim': 'EPTA',
		'JBO.ROACH.1520.tim': 'EPTA',
		'LEAP.1396.tim': 'EPTA',
		'NRT.BON.1400.tim': 'EPTA',
		'NRT.BON.1600.tim': 'EPTA',
		'NRT.NUPPI.1484.tim': 'EPTA',
		'NRT.NUPPI.1854.tim': 'EPTA',
		'SRT.DFB.1400':  'EPTA',
		'SRT.DFB.330':  'EPTA',
		'SRT.ROACH.1400': 'EPTA',
		'SRT.ROACH.330': 'EPTA',
		'WSRT.P1.1380.2.tim': 'EPTA',
		'WSRT.P1.1380.C.tim': 'EPTA',
		'WSRT.P1.1380.tim': 'EPTA',
		'WSRT.P1.2273.2.tim': 'EPTA',
		'WSRT.P1.2273.C.tim': 'EPTA',
		'WSRT.P1.2273.tim': 'EPTA',
		'WSRT.P1.323.tim': 'EPTA',
		'WSRT.P1.328.tim': 'EPTA',
		'WSRT.P1.367.tim': 'EPTA',
		'WSRT.P1.382.tim': 'EPTA',
		'WSRT.P2.1380.tim': 'EPTA',
		'WSRT.P2.2273.tim': 'EPTA',
		'WSRT.P2.350.tim': 'EPTA',
		'GM_GWB_1460_100_b1_pre36.tim': 'InPTA',
		'GM_GWB_1460_200_b0_post36.tim': 'InPTA',
		'GM_GWB_500_100_b1_pre36.tim':  'InPTA',
		'GM_GWB_500_200_b1_post36.tim': 'InPTA',
        'MK.LBAND.1420.tim': 'MeerTime'}

# Additional MeerKAT jump labels (as in notebook)
jumps_per_system['MK.LBAND.1420'] = ["MK.LBAND.{:0.0f}".format(meerkat['band_top'] - meerkat['bw']*(int(x) + .5)) for x in range(meerkat['nchan']) ]#np.linspace(778,1650,16)]

# Common comment/skip tokens used when editing .tim files
TIM_HEADER_KEYS = ('TIME', 'MODE', 'FORMAT', '-padd')
TIM_OTHER_SKIP_KEYS = ('below', 'SKIP', 'EFAC')

def _backup_and_replace_tim(tim_path: Path) -> None:
    """Create .orig backup and replace original with .new (notebook-style)."""
    orig = tim_path.with_suffix(".orig")
    new = tim_path.with_suffix(".new")
    if not new.exists():
        raise FileNotFoundError(f"Expected new file {new} not found")
    shutil.copy2(tim_path, orig)
    shutil.move(str(new), str(tim_path))

def get_psrs(homepath):
    """Return pulsar directory names under the dataset root.

    Args:
        homepath: Dataset root containing JXXXX+XXXX directories.

    Returns:
        Sorted list of pulsar directory names.
    """
    return sorted([x.split('/')[-1] for x in glob(os.path.join(homepath,'J?????????'))])

def get_timfiles(homepath, psrs):
    """Return per-pulsar timfile basenames.

    Args:
        homepath: Dataset root containing pulsar directories.
        psrs: Iterable of pulsar names.

    Returns:
        List of lists of timfile basenames for each pulsar.
    """
    timfiles=[]
    for psr in psrs:
        timfiles.append(sorted([x.split('/')[-1] for x in glob(os.path.join(homepath,psr,'tims/*.tim')) if "NRT.NUXPI" not in x]))
    return timfiles

def fix_bad_padd_lines(timfilepath, repo):
    """Fix malformed '-padd' line breaks in a tim file.

    Args:
        timfilepath: Path to a .tim file.
        repo: GitPython repository used for commits.

    Returns:
        None.
    """
    with open(timfilepath, 'r') as readfile:
        with open(timfilepath.replace('.tim', '.paddfixed'), 'w') as writefile:
            for i, line in enumerate(readfile):
                # Get rid of the trailing newline (if any).
                line = line.rstrip("\n")
                line = line.rstrip(" ")
                if "\n -padd" in line:
                    line = oldline + line.replace("\n -padd", " -padd")
                    print(line, file=writefile)
                elif "\n-padd" in line:
                    line = oldline + line.replace("\n-padd", " -padd")
                    print(line, file=writefile)
                else:
                    oldline = line

    print("mv {} {}".format(timfilepath.replace('.tim', '.paddfixed'), timfilepath))
    os.system("mv {} {}".format(timfilepath.replace('.tim', '.paddfixed'), timfilepath))
    repo.index.add(timfilepath)
    repo.index.commit("Fixed bad padd lines in {}".format(timfilepath), author=author, committer=committer)
    return None

def fix_timfiles(homepath, psr, repo):
    """Apply all timfile cleanup steps for a pulsar.

    Args:
        homepath: Dataset root for pulsar directories.
        psr: Pulsar name (e.g., JXXXX+XXXX).
        repo: GitPython repository used for commits.

    Returns:
        None.
    """
    for timfilepath in glob(os.path.join(homepath, psr, 'tims', "*.tim")):
        with open(timfilepath, 'r') as readfile:
            with open(timfilepath.replace('.tim', '.tabfixed'), 'w') as writefile:
                for i, line in enumerate(readfile):
                    # Get rid of the trailing newline (if any).
                    line = line.rstrip("\n")
                    line = line.rstrip(" ")
                    line = line.replace("\t", " ")
                    line = line.replace("  ", " ")
                    print(line, file=writefile)
        foundpaddlines=skiptimlines(timfilepath, my_string="\n-padd")
        if len(foundpaddlines)>0:
            print("{} has bad padd lines. Fixing.".format(filepath))
            fix_bad_padd_lines(timfilepath, repo)

        print("mv {} {}".format(timfilepath.replace('.tim', '.tabfixed'), timfilepath))
        os.system("mv {} {}".format(timfilepath.replace('.tim', '.tabfixed'), timfilepath))
        repo.index.add(timfilepath)
        repo.index.commit("Cleaned {}".format(timfilepath), author=author, committer=committer)
    return None

def _make_gen(reader):
    """Yield raw file chunks from a binary reader.

    Args:
        reader: Callable that returns bytes when called.

    Yields:
        Raw byte chunks from the reader.
    """
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    """Count newline characters in a binary file.

    Args:
        filename: Path to the file.

    Returns:
        Number of newline bytes found.
    """
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )

def findlines(searchstring, filename):
    """Find line numbers containing a substring.

    Args:
        searchstring: Substring to match.
        filename: Path to the file.

    Returns:
        List of 1-based line numbers containing the substring.
    """
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f, 1):
            if searchstring in line:
                lines.append(i)
    return lines

def findblanklines(filename):
    """Return zero-based indices of blank lines.

    Args:
        filename: Path to the file.

    Returns:
        List of zero-based indices for blank lines.
    """
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                lines.append(i - 1)
    return lines

def findlinesstartingwith(searchstring, filename, ret_arr=False):
    """Find line numbers that start with a prefix.

    Args:
        searchstring: Prefix to match at line start.
        filename: Path to the file.
        ret_arr: If True, include the first two tokens of each matched line.

    Returns:
        List of line numbers (1-based), or a list of [lineno, token1, token2] rows when ``ret_arr`` is True.
    """
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f, 1):
            if line.startswith(searchstring) and ret_arr:
                _ll = line.split()[:2]
                _ll[:0] = [i]
                lines.append(_ll)
            elif line.startswith(searchstring):
                lines.append(i)
    return lines

def skiptimlines(timfilename,
                 my_string="FORMAT",
                 only_once=False,
                 ret_arr=False):
    """Compute zero-based row indices to skip based on a prefix match.

    Args:
        timfilename: Path to the tim file.
        my_string: Line prefix to match.
        only_once: If True, limit to the first match.
        ret_arr: If True, return both skip rows and the matched line array.

    Returns:
        List of zero-based row indices, or ``(skiprows, lines)`` when ``ret_arr`` is True.
    """
    lines = np.array(findlinesstartingwith(searchstring=my_string, filename=timfilename, ret_arr=ret_arr))
    if len(lines.shape) > 1:
        foundlines = list(lines[:,0])
    if len(lines.shape) == 1:
        foundlines = list(lines)
    try:
        foundlines = [int(x) - 1 for x in foundlines]
    except TypeError as e:
        print(f"{e} because {foundlines}")
    try:
        skiprows = list(foundlines)
    except Exception as e:
        print('No lines to skip.')
        skiprows = []
    if ret_arr:
        return skiprows, lines
    else:
        return skiprows

def skiplines(filename,
              offset=0,
              my_string="---------------------------------------------------------------------------------------------------",
              only_once=False,
             quiet=True):
    """Compute row indices to skip after a delimiter line.

    Args:
        filename: Path to the file.
        offset: Index offset applied to matched lines.
        my_string: Delimiter string to search for.
        only_once: If True, only consider the first delimiter.
        quiet: If False, print warnings when delimiters are missing.

    Returns:
        List of zero-based row indices to skip.
    """
    foundlines = findlines(searchstring=my_string, filename=filename)
    try:
        skiprows = [] #list(np.arange(0,foundlines[0]+offset,1))
        if not only_once:
            try:
                if not my_string == "C":
                    skiprows += list(np.arange(foundlines[1]-1+offset,rawgencount(filename),1))
            except Exception as e:
                if not quiet:
                    print('Didn\'t find \'%s\' more than once. %s'%(my_string,e))
                pass
    except Exception as e:
        skiprows = []
        if not quiet:
            print('No lines to skip.')
    return skiprows

def skip_blank_lines(timfilename):
    """Return zero-based indices for blank lines in a tim file.

    Args:
        timfilename: Path to the tim file.

    Returns:
        List of zero-based row indices.
    """
    try:
        foundlines = findlinesstartingwith(searchstring="\n", timfilename=timfilename)
        foundlines = [int(x) - 1 for x in foundlines]
        skiprows = list(foundlines)
    except Exception as e:
        print('No blank lines.')
        skiprows = []
    return skiprows

def convert_numeric_cols(dmf, start=1):
    """Coerce TOA columns to float and prune implausible rows.

    Args:
        dmf: DataFrame parsed from a tim file.
        start: Column index for the first numeric column (freq).

    Returns:
        DataFrame with numeric columns converted and rows filtered.
    """
    dmf[start] = dmf[start].astype(np.float64)
    dmf[start+1] = dmf[start+1].astype(np.float64)
    dmf[start+2] = dmf[start+2].astype(np.float64)
    dmf = dmf[dmf[start+1] > 10000.] #Forced pruning to remove unusual lines
    return dmf

def fix_my_df(dmf, bad_tim_lines):
    """Normalize a parsed timfile DataFrame after read errors.

    Args:
        dmf: DataFrame parsed from a tim file.
        bad_tim_lines: Mapping of malformed line keys to replacement values.

    Returns:
        Fixed DataFrame with numeric columns coerced.
    """
    while len(dmf.columns) == 2:
        if dmf.keys().values[0] in bad_tim_lines.keys():
            dmf = two_column_fix(dmf,
                                 keyword=dmf.keys().values[0],
                                 value=str(bad_tim_lines.get(dmf.keys().values[0])))
    if len(dmf.columns) > 5:
        dmf.dropna(inplace=True, how='all')
        dmf.columns = list(map(lambda x: x, range(len(dmf.columns))))
        try:
            dmf = convert_numeric_cols(dmf, start=1)
        except ValueError as e:
            print(dmf[1])
            dmf = convert_numeric_cols(dmf, start=2)
    return dmf

def drop_my_cols(df, string):
    """Drop a column that contains a specific sentinel string.

    Args:
        df: DataFrame to mutate in place.
        string: Sentinel value identifying the column.

    Returns:
        None.
    """
    n = (df.stack() == string).idxmax()[1]
    df.columns.get_loc(n)
    df.drop([n], axis=1, inplace=True)
    return None

def find_flag_column(df, flag_key="-sys"):
    """Locate the column index for a given timfile flag.

    Args:
        df: Parsed timfile DataFrame.
        flag_key: Flag token to search for (e.g., ``-sys``).

    Returns:
        Column index if found, otherwise None.
    """
    try:
        col_num = np.unique(np.where(df.to_numpy()==flag_key)[1])[0]
        return col_num
    except Exception as e:
        print("{} has no matching column".format(flag_key))
        return None

def strip_leading_whitespace(timfile):
    """Legacy stub for stripping leading whitespace via ``sed``.

    Args:
        timfile: Path to the tim file.

    Returns:
        None.

    Notes:
        The current implementation invokes ``sed`` without a target file and does
        not modify the input.
    """
    os.system("sed -i\".orig\" ")
    return None

def skipper(timfilepath):
    """Compute a consolidated skip-row list for tim parsing.

    Args:
        timfilepath: Path to the tim file.

    Returns:
        Sorted list of zero-based row indices to skip.
    """
    rowstoskip=[0]
    badstarts=["TIME", "MODE", "FORMAT", "-padd", "Er", "C\ ", "C ", "CC", "end", "#", "c\ ", "c "]
    badstarts=badstarts.extend([" "+x for x in badstarts])
    print(badstarts)
    for searchstring in badstarts:
        rowstoskip.extend(skiptimlines(timfilepath, my_string=searchstring))
    other_skip_keys = ['below', 'SKIP', 'EFAC']
    for otherstring in other_skip_keys:
        rowstoskip.extend(skiplines(timfilepath, my_string=otherstring))
    rowstoskip.extend(findblanklines(timfilepath))
    time_statements_rowidx,time_skip_array = skiptimlines(timfilepath,
                                                           my_string="TIME",
                                                           ret_arr=True)
    rowstoskip.extend(time_statements_rowidx)
    return sorted(rowstoskip)

def read_tamfile(timfile, separator=r'\s+', rowstoskip=1, time_skip_array=None, error=False, warning=False, **kwargs):
    """Read a tim file into a DataFrame with legacy skip/repair rules.

    Args:
        timfile: Path to the tim file.
        separator: Regex separator passed to ``pandas.read_csv``.
        rowstoskip: Row indices to skip or an integer count.
        time_skip_array: Array of TIME statements for offset handling.
        error: If True, raise on failure.
        warning: If True, print a warning on failure.
        **kwargs: Extra keyword arguments forwarded to ``read_csv``.

    Returns:
        DataFrame of parsed rows (may be empty on failure).
    """
    if type(rowstoskip) == int:
        rowstoskip=list(range(rowstoskip))
    badstarts=["MODE", "FORMAT", "-padd", "Er", "C\ ", "C ", "CC", "end", "#", "c\ ", "c "]
    badstarts.extend([" "+x for x in badstarts])

    for searchstring in badstarts:
        rowstoskip.extend(skiptimlines(timfile, my_string=searchstring))
    bad_tim_lines={'FORMAT':'1', 'TIME':'0', 'MODE':'2'}
    #bad_lines dict to grab improperly read tim timfiles
    if time_skip_array.size == 2:
        bad_tim_lines['TIME'] = str(time_skip_array[0,2])
    other_skip_keys = ['below', 'SKIP', 'EFAC']
    other_skip_keys.extend(list(bad_tim_lines.keys()))
    for otherstring in other_skip_keys:
        rowstoskip.extend(skiplines(timfile, my_string=otherstring))
    rowstoskip.extend(findblanklines(timfile))
    time_statements_rowidx,time_skip_array = skiptimlines(timfile,
                                                       my_string="TIME",
                                                       ret_arr=True)
    rowstoskip.extend(time_statements_rowidx)
    rowstoskip = sorted(rowstoskip)
    keepcrows=None
    for searchstring in ["c{}".format(x) for x in range(9)]:
        keepcrows = list(set(skiptimlines(timfile, my_string=searchstring)))
        rowstoskip = [x for x in rowstoskip if x not in keepcrows]

    try:
        dmf = pd.read_csv(timfile,engine='python',sep=separator,header=None,
                          skiprows=rowstoskip,warn_bad_lines=False,
                          error_bad_lines=False,skip_blank_lines=True,
                          converters={0: str.strip}
             )

        if "-padd" in list(dmf[0]):
            print("Ugh. Found padds on newlines.")
            rowstoskip = list(set(rowstoskip +[x+1 for x in rowstoskip]))
            dmf = pd.read_csv(timfile,engine='python',sep=separator,header=None,
                              skiprows=rowstoskip,warn_bad_lines=False,
                              error_bad_lines=False,skip_blank_lines=True,
                              converters={0: str.strip}
                 )
            padds = dmf[dmf[0].isin(["-padd"])]
            rest = dmf[~dmf[0].isin(["-padd"])]
            dmf = pd.concat([rest,padds.set_index(rest.index)], axis=1)
            dmf.dropna(axis=1,inplace=True, how='all')

        # for false_flag in ["-flag", "-pn"]:
        #     if false_flag in dmf:
        #         print("Found false flag")
        #         drop_my_cols(dmf, false_flag)
        dmf = fix_my_df(dmf, bad_tim_lines)
        for i in dmf.columns:
            try:
                if dmf[i].dtype == 'object':
                    dmf[i] = pd.to_numeric(dmf[i], errors='ignore')
            except Exception as e:
                print("Failed with error {} whose arguments are {}".format(e,e.args))
        return dmf
    except Exception as e:
        print(e)
        if error:
            raise SystemExit()
        if warning:
            print('Tamfile failed to read %s with error\n%s'%(timfile,e))
        return pd.DataFrame()

def get_psr_info(homepath, psrs=None, timfiles=None, repo=None):
    """Collect per-pulsar flag metadata from tim files.

    Args:
        homepath: Dataset root directory.
        psrs: Optional list of pulsar names.
        timfiles: Optional list of timfile lists per pulsar.
        repo: GitPython repository handle (required).

    Returns:
        Nested dict mapping pulsar -> timfile -> flag metadata.
    """
    if not repo:
        print("Repo must be supplied!")
        raise GitCommandError
    psr_info = {}
    if not psrs and not timfiles:
        psrs = get_psrs(homepath)
        timfiles = get_timfiles(homepath, psrs)
    for psr, timfileset in zip(psrs,timfiles):
        timfile_info = {}
        print("Procesing timfiles for {}".format(psr))
        for timfile in timfileset:
            filepath=os.path.join(homepath, psr, 'tims', timfile)
            time_statements_rowidx,time_skip_array = skiptimlines(filepath,
                                                                   my_string="TIME",
                                                                   ret_arr=True)
            try:
                df = read_tamfile(filepath, time_skip_array=time_skip_array)
            except Exception as e:
                print(e)
                # try:
                #     df = read_tamfile(filepath, time_skip_array=time_skip_array, separator=r'\t')
                # except Exception as e:
                #     print(e)
                df = pd.DataFrame()

            key_collector = {}
            if not df.empty:
                for flag in ["-sys", "-pta", "-group", "-be"]:
                    col_num = find_flag_column(df, flag_key=flag)
                    if col_num:
                        test = sorted(np.unique(df[col_num+1].values))
                        if len(test) > 1:
                            test = [ a for a in test if not a.isnumeric() ]
                        key_collector[flag] = test
                    else:
                        key_collector[flag] = 'missing'
            else:
                print("No data read in for {}".format(timfile))
            timfile_info[timfile] = key_collector
        psr_info[psr] = timfile_info
    return psr_info

def show_psr_info(psr_info, psr=None):
    """Print pulsar info as formatted JSON.

    Args:
        psr_info: Nested dict from :func:`get_psr_info`.
        psr: Optional pulsar name to print.

    Returns:
        None.
    """
    if psr:
        print(json.dumps(psr_info[psr], indent=4, default=str))
    else:
        print(json.dumps(psr_info, indent=4, default=str))

def isfloat(my_object):
    """Return True if the object can be converted to float.

    Args:
        my_object: Value to test.

    Returns:
        True if float conversion succeeds.
    """
    try:
        float(my_object)
        return True
    except:
        return False

def ismatched(freq, sat, keep_freq, keep_mjd, time_window, freq_range):
    """Check whether a TOA matches a reference frequency/time window.

    Args:
        freq: Candidate frequency.
        sat: Candidate MJD.
        keep_freq: Reference frequency.
        keep_mjd: Reference MJD.
        time_window: Allowed time window (days).
        freq_range: Allowed frequency range (MHz).

    Returns:
        True if the candidate is within both windows.
    """
    try:
        if (isfloat(freq) and (freq >= keep_freq - freq_range and
                               freq <= keep_freq + freq_range) and
            isfloat(sat) and (sat >= keep_mjd - time_window and
                              sat <= keep_mjd + time_window)):
            return True
        else:
            return False
    except Exception as e:
        print("Warning: Matching failed with error {}".format(e))
        return False

def cleanline(line):
    """Strip newline and surrounding whitespace from a line.

    Args:
        line: Input line string.

    Returns:
        Cleaned line string.
    """
    line = line.rstrip("\n")
    line = line.rstrip(" ")
    line = line.lstrip(" ")
    return line

def freqsat(line):
    """Extract frequency and MJD from a timfile line.

    Args:
        line: Timfile line string.

    Returns:
        Tuple ``(freq, mjd)`` as floats, or ``(None, None)`` on failure.
    """
    try:
        line = line.split()
        if isfloat(line[1]) and isfloat(line[2]):
            return float(line[1]), float(line[2])
        else:
            return None, None
    except Exception as e:
        #print("Warning: {}".format(e))
        return None, None

def selectfreqsat(_line, _timfile, freqfrom):
    """Select frequency/MJD based on timfile naming or TOA line.

    Args:
        _line: Timfile line string.
        _timfile: Timfile basename.
        freqfrom: Either ``'group'`` or ``'toa'``.

    Returns:
        Tuple ``(freq, mjd)`` as floats.
    """
    _freq, _sat = freqsat(_line)
    if freqfrom == 'group':
        if 'LEAP'  in _timfile:
            _freq = float(_timfile.split('.')[1])
        else:
            _freq = float(_timfile.split('.')[2])
    elif freqfrom == 'toa':
        pass
    else:
        print("Warning: freqfrom can only be group or toa, not {}. Using toa.".format(freqfrom))
    return _freq, _sat

def comment_overlapped_toa(homepath, psr, retain_backend, drop_backends, time_window, freqfrom='toa'):
    """Comment out TOAs that overlap a retained backend.

    Args:
        homepath: Dataset root directory.
        psr: Pulsar name.
        retain_backend: Timfile basename to retain.
        drop_backends: Iterable of timfile basenames to modify.
        time_window: Matching time window (days).
        freqfrom: Source for frequency comparison (``'group'`` or ``'toa'``).

    Returns:
        None.
    """
    freq_range = backend_bw.get(retain_backend)
    #print("Keeping TOAs {}".format(retain_backend))
    with open(os.path.join(homepath,psr,'tims/',retain_backend), 'r') as keeptim:
        keeptimskiprows = skipper(os.path.join(homepath,psr,'tims/',retain_backend))
        for i, keepline in enumerate(keeptim):
            keepline = cleanline(keepline)
            if i not in keeptimskiprows:
                keep_freq, keep_sat = selectfreqsat(keepline, retain_backend, freqfrom)
                #print("{}-{}-{}-{}-{}-{}".format(keep_freq, keep_sat, 'mod_freq', 'mod_sat', time_window, freq_range))

                for timfile in drop_backends:
                    #print("Modifying {} for {}".format(timfile, psr))
                    if os.path.exists(os.path.join(homepath,psr,'tims/',timfile)):
                        modtimskiprows = skipper(os.path.join(homepath,psr,'tims/',timfile))
                        mbw = backend_bw.get(timfile)
                        with open(os.path.join(homepath,psr,'tims/',timfile), 'r') as modtim:
                            with open(os.path.join(homepath,psr,'tims/',timfile.replace(".tim", "_mod.tim")), 'w') as newfile:

                                if keep_freq and keep_sat:
                                            for j, modline in enumerate(modtim):
                                                modline = cleanline(modline)
                                                mod_freq, mod_sat = selectfreqsat(modline, timfile, freqfrom)
                                                myline = ""

                                                if j not in modtimskiprows and len(modline.split()) > 2 and isfloat(mod_freq) and ismatched(mod_freq, mod_sat,
                                                                                                                                            keep_freq, keep_sat,
                                                                                                                                            time_window, freq_range+mbw):
                                                    #print("Found match {}:{} PSR {} on {}:{}".format(retain_backend, timfile, psr, keep_sat, mod_sat))
                                                    myline="C {}{}".format(modline,os.linesep)
                                                    print(myline, file=newfile)
                                                else:
                                                    #Print unmatched lines
                                                    print(modline, file=newfile)
                                            os.system("\cp -f {}/{}/tims/{} {}/{}/tims/{}".format(homepath, psr, timfile, homepath, psr, timfile.replace(".tim", ".orig")))
                                            os.system("mv {}/{}/tims/{} {}/{}/tims/{}".format(homepath, psr, timfile.replace(".tim", "_mod.tim"), homepath, psr, timfile))

    return None

def remove_overlap(psr_info, homepath, overlapped_timfiles, time_window):
    """Remove overlapping TOAs for known overlapping backend pairs.

    Args:
        psr_info: Pulsar info dict.
        homepath: Dataset root directory.
        overlapped_timfiles: Mapping of retained backend to overlapped backends.
        time_window: Matching time window (days).

    Returns:
        None.
    """
    for psr in tqdm(psr_info.keys()):
        for retain_backend in overlapped_timfiles.keys():
            if retain_backend in sorted(psr_info[psr].keys()):
                drop_backends = overlapped_timfiles.get(retain_backend)
                #if any([x in sorted(psr_info[psr].keys()) for x in drop_backends]):
                comment_overlapped_toa(homepath, psr, retain_backend, drop_backends, time_window, freqfrom='group')


    return None

def remove_nuppi_big(parfile, timfile):
    """Remove NRT.NUPPI/NUXPI entries from par and tim files.

    Args:
        parfile: Path to the .par file.
        timfile: Path to the ``*_all.tim`` file.

    Returns:
        None.
    """
    with open(parfile, 'r') as nuppioldpar, open(timfile, 'r') as nuppioldtim:
        with open(parfile.replace('.par', '.new'), 'w') as nuppinewpar, open(timfile.replace('_all.tim', '_all.new'), 'w') as nuppinewtim:

            for parline in nuppioldpar:
                parline = parline.rstrip("\n")
                parline = parline.rstrip(" ")
                if "NRT.NUPPI." in parline or "NRT.NUXPI." in parline:
                    print("Removing old NUPPI lines from {}".format(parfile))
                else:
                    print(parline, file=nuppinewpar)

            for timline in nuppioldtim:
                if "NRT.NUPPI." in timline or "NRT.NUXPI." in timline:
                    print("Removing old NUPPI lines from {}".format(timfile))
                else:
                    print(timline, file=nuppinewtim)
    return None

def remove_nuppi_fromParTimfiles(homepath, psr_info, use_newfile=False):
    """Apply NUPPI removal across all pulsars.

    Args:
        homepath: Dataset root directory.
        psr_info: Pulsar info dict.
        use_newfile: If True, replace originals with .new outputs.

    Returns:
        None.
    """
    for psr in psr_info.keys():
        parfile = psr+".par"
        timfile = psr+"_all.tim"
        try:
            remove_nuppi_big(os.path.join(homepath,psr,parfile),os.path.join(homepath,psr,timfile))
            if use_newfile:
                os.system("\cp -f {}/{}/{} {}/{}/{}".format(homepath, psr, parfile, homepath, psr, parfile.replace(".par", ".orig")))
                os.system("mv {}/{}/{} {}/{}/{}".format(homepath, psr, parfile.replace(".par", ".new"), homepath, psr, parfile))
        except Exception as e:
            print(e)
            pass


    return None

def split_on_date(timfilepath, mjd_cut=57500.):
    """Split a NUPPI timfile into two files around an MJD cutoff.

    Args:
        timfilepath: Path to the NUPPI timfile.
        mjd_cut: MJD cutoff separating NUPPI1 and NUPPI2.

    Returns:
        None.

    Notes:
        This function also commits the change using the global ``repo`` handle.
    """
    rowstoskip=[0]
    for searchstring in ["TIME", "MODE", "FORMAT", "-padd"]:
        rowstoskip.extend(skiptimlines(timfilepath, my_string=searchstring))
    other_skip_keys = ['below', 'SKIP', 'EFAC']
    for otherstring in other_skip_keys:
        rowstoskip.extend(skiplines(timfilepath, my_string=otherstring))
    rowstoskip.extend(findblanklines(timfilepath))
    time_statements_rowidx, time_skip_array = skiptimlines(timfilepath,
                                                       my_string="TIME",
                                                       ret_arr=True)

    rowstoskip.extend(time_statements_rowidx)
    rowstoskip = sorted(rowstoskip)
    with open(timfilepath, 'r') as workingfile:
        with open(timfilepath.replace("NUPPI","NUPPI1"), 'w') as nuppiold, open(timfilepath.replace("NUPPI","NUPPI2"), 'w') as nuppinew:
            for i, line in enumerate(workingfile):
                # Get rid of the trailing newline (if any).
                line = line.rstrip("\n")
                line = line.rstrip(" ")
                if i in rowstoskip:
                    if "\n -padd" in line:
                        line = line.replace("\n -padd", " -padd")
                    elif "FORMAT" in line or 'MODE' in line:
                        print(line, file=nuppiold)
                        print(line, file=nuppinew)
                    else:
                        pass
                elif mjd_cut > float(line.split()[2]):
                    print(line.replace('NUPPI', 'NUPPI1'), file=nuppiold)
                elif mjd_cut <= float(line.split()[2]):
                    print(line.replace('NUPPI', 'NUPPI2'), file=nuppinew)
                else:
                    print(line.split())
                    pass
    os.system("mv {} {}".format(timfilepath, timfilepath.replace('NUPPI', 'NUXPI')))
    repo.index.add([timfilepath.replace("NUPPI","NUPPI1"),timfilepath.replace("NUPPI","NUPPI2")])
    repo.index.commit("Adding NUPPI split tim files for {}".format(timfilepath.split('/')[-3]))
    return None

def split_nuppi(psr_info, homepath):
    """Split NUPPI timfiles for pulsars with NUPPI group flags.

    Args:
        psr_info: Pulsar info dict.
        homepath: Dataset root directory.

    Returns:
        None.
    """
    for psr, timfile in psr_info.items():
        print("Working on {}".format(psr))
        for timfile, flags in timfile.items():
            for flag, value in flags.items():
                if 'NUPPI' in value[0] and flag == '-group':
                    print("Using timfile : {}".format(timfile))
                    split_on_date(os.path.join(homepath, psr, 'tims', timfile), mjd_cut=57600)

def insert_missing_flags(timfilepath, flag, value, ):
    """Append a flag/value pair to each TOA line in a timfile.

    Args:
        timfilepath: Path to the timfile.
        flag: Flag token to insert (e.g., ``-be``).
        value: Flag value to insert.

    Returns:
        None.
    """
    print("Adding {} {} to {}".format(flag, value, timfilepath))
    rowstoskip=[0]
    for searchstring in ["TIME", "MODE", "FORMAT", "-padd"]:
        rowstoskip.extend(skiptimlines(timfilepath, my_string=searchstring))
    other_skip_keys = ['below', 'SKIP', 'EFAC']
    for otherstring in other_skip_keys:
        rowstoskip.extend(skiplines(timfilepath, my_string=otherstring))
    rowstoskip.extend(findblanklines(timfilepath))
    time_statements_rowidx, time_skip_array = skiptimlines(timfilepath,
                                                       my_string="TIME",
                                                       ret_arr=True)

    rowstoskip.extend(time_statements_rowidx)
    rowstoskip = sorted(rowstoskip)
    with open(timfilepath, 'r') as workingfile:
        with open(timfilepath.replace(".tim",".new"), 'w') as newfile:
            for i, line in enumerate(workingfile):
                # Get rid of the trailing newline (if any).
                line = line.rstrip("\n")
                line = line.rstrip(" ")
                if i in rowstoskip:
                    if "\n -padd" in line:
                        line = line.replace("\n -padd", " -padd")
                    else:
                        pass
                else:
                    line += " {} {}".format(flag, value)
                print(line, file=newfile)
    return None

def insert_system_flags(timfilepath, flag, value, ):
    """Append system flags inferred from frequency to each TOA line.

    Args:
        timfilepath: Path to the timfile.
        flag: Flag token to insert (typically ``-sys``).
        value: Unused placeholder; computed per line.

    Returns:
        None.
    """
    print("Adding {} {} to {}".format(flag, value, timfilepath))
    rowstoskip=[0]
    for searchstring in ["TIME", "MODE", "FORMAT", "-padd"]:
        rowstoskip.extend(skiptimlines(timfilepath, my_string=searchstring))
    other_skip_keys = ['below', 'SKIP', 'EFAC']
    for otherstring in other_skip_keys:
        rowstoskip.extend(skiplines(timfilepath, my_string=otherstring))
    rowstoskip.extend(findblanklines(timfilepath))
    time_statements_rowidx, time_skip_array = skiptimlines(timfilepath,
                                                       my_string="TIME",
                                                       ret_arr=True)

    rowstoskip.extend(time_statements_rowidx)
    rowstoskip = sorted(rowstoskip)
    freqs = []
#     tel = timfilepath.split('/')[-1].split('.')[0]
#     be = timfilepath.split('/')[-1].split('.')[1]
    with open(timfilepath, 'r') as workingfile:
        with open(timfilepath.replace(".tim",".new"), 'w') as newfile:
            for i, line in enumerate(workingfile):
                # Get rid of the trailing newline (if any).
                line = line.rstrip("\n")
                line = line.rstrip(" ")
                if i in rowstoskip:
                    if "\n -padd" in line:
                        line = line.replace("\n -padd", " -padd")
                    else:
                        pass
                else:
                    _wfreq = float(line.lstrip(" ").split(' ')[1])
                    _jumps = jumps_per_system[timfilepath.split('/')[-1].rstrip('.tim')]
                    _temp = np.array([float(x.split('.')[-1]) for x in _jumps])
                    _flabel = int(_temp[np.abs(_temp - _wfreq).argmin()])
                    value = "{}.{}".format(timfilepath.split('/')[-1].rstrip('.tim').rsplit('.',1)[0], _flabel)
                    line += " {} {}".format(flag, value)

                print(line, file=newfile)
#             print("\n----------------\n", max(freqs), min(freqs), "\n----------------\n")
#             #fig, ax = plt.subplots(figsize=(10, 20))
#             #ax.hist(freqs, bins=16)#801, orientation='horizontal')
    return None

def savenewfile(homepath, psr, timfile):
    """Notebook-compatible wrapper that backs up and replaces a .tim using .new."""
    tim_path = Path(homepath) / psr / "tims" / timfile
    _backup_and_replace_tim(tim_path)
    print(f"Replaced {tim_path.name} (backup at {tim_path.with_suffix('.orig').name})")
    return 0


def update_timfiles(psr_info, homepath, magic_add_sysflag=None):
    """Update timfiles by inserting missing flags.

    Args:
        psr_info: Pulsar info dict from :func:`get_psr_info`.
        homepath: Dataset root directory.
        magic_add_sysflag: If True, attempt per-TOA system flags.

    Returns:
        None.
    """
    for psr, timfile in psr_info.items():
        for timfile, flags in timfile.items():
            newfile = False
            for flag, value in flags.items():
                if value == 'missing' and flag == '-sys':
                    print("WARNING: For PSR {} {} has missing system flags! Check that file is proper.".format(psr, timfile))
                    if magic_add_sysflag:
                        insert_system_flags(os.path.join(homepath, psr, 'tims', timfile),
                                             flag, timfile.rstrip(".tim"))
                        savenewfile(homepath, psr, timfile)

                if value == 'missing' and flag == '-be':
                    insert_missing_flags(os.path.join(homepath, psr, 'tims', timfile),
                                         flag, timfile.split(".")[1])
                    savenewfile(homepath, psr, timfile)

                if value == 'missing' and flag == '-pta':
                    insert_missing_flags(os.path.join(homepath, psr, 'tims', timfile), flag, pta_systems.get(timfile))
                    savenewfile(homepath, psr, timfile)

                if value == 'missing' and flag == '-group':
                    insert_missing_flags(os.path.join(homepath, psr, 'tims', timfile),
                                         flag, timfile.rstrip(".tim"))
                    savenewfile(homepath, psr, timfile)

    return None

def insert_missing_timfiles(psr, alltimfilepath, psr_info):
    """Append missing INCLUDE lines to a pulsar's ``*_all.tim`` file.

    Args:
        psr: Pulsar name.
        alltimfilepath: Path to ``*_all.tim``.
        psr_info: Pulsar info dict.

    Returns:
        None.
    """
    addtimfiles = [x for x in psr_info.get(psr)]
    with open(alltimfilepath, 'r') as workingfile:
        with open(alltimfilepath.replace("_all.tim","_all.new"), 'w') as newfile:
            for line in workingfile:
                line = line.rstrip(os.linesep)
                line = line.rstrip(" ")
                line_contents = line.split()
                if line.strip() and ("INCLUDE" == line_contents[0]) and (line_contents[1] in addtimfiles):
                    addtimfiles.remove(line_contents[1])
    with open(alltimfilepath.replace(".tim",".new"), 'a') as newfile:
        for newtims in addtimfiles:
            goodlines = (rawgencount(os.path.join(alltimfilepath.rpartition('/')[0],'tims',newtims)) - len(skipper(os.path.join(alltimfilepath.rpartition('/')[0],'tims',newtims))))
            if goodlines > 9:
                line = "INCLUDE tims/{}".format(newtims)
                print(line, file=newfile)
                print(line)
            else:
                print("Dropping {} for {} because it has only {} TOAs".format(newtims, psr, goodlines))
                #newfile.write(line)

    return None

def update_alltims(homepath, psr_info, use_newfile=False):
    """Update all ``*_all.tim`` files with missing INCLUDE lines.

    Args:
        homepath: Dataset root directory.
        psr_info: Pulsar info dict.
        use_newfile: If True, replace originals with .new outputs.

    Returns:
        None.
    """
    for psr in psr_info.keys():
        alltim = psr+"_all.tim"
        try:
            insert_missing_timfiles(psr, alltimfilepath=os.path.join(homepath, psr, alltim), psr_info=psr_info)
            if use_newfile:
                os.system("\cp -f {}/{}/{} {}/{}/{}".format(homepath, psr, alltim, homepath, psr, alltim.replace("_all.tim", "_all.orig")))
                os.system("mv {}/{}/{} {}/{}/{}".format(homepath, psr, alltim.replace("_all.tim", "_all.new"), homepath, psr, alltim))
        except Exception as e:
            print(e)
            pass

    return None

def flatten(xss):
    """Flatten a list of iterables into a single list.

    Args:
        xss: Iterable of iterables.

    Returns:
        Flattened list of items.
    """
    flat_list = []
    for xs in xss:
        if xs:
            for x in xs:
                flat_list.append(x)
    return flat_list

def insert_missing_jumps(psr, parfilepath, psr_info):
    """Normalize EPHEM/CLK/NE_SW lines in a .par file.

    Args:
        psr: Pulsar name (unused).
        parfilepath: Path to the .par file.
        psr_info: Pulsar info dict (unused).

    Returns:
        None.
    """
    with open(parfilepath, 'r') as workingfile:
        with open(parfilepath.replace(".par",".new"), 'w') as newfile:
            for line in workingfile:
                line = line.rstrip("\n")
                line = line.rstrip(" ")
                line_contents = line.split()
                if "EPHEM" in line_contents and not(line_contents[1] == ephem_global):
                    line = line.replace(line_contents[1], ephem_global)
                if "CLK" == line_contents[0] and not(line_contents[1] == bipm_version):
                    line = line.replace(line_contents[1], bipm_version)
                if "NE_SW" in line_contents and not(line_contents[1] == nesw_value):
                    line = line.replace(line_contents[1], nesw_value)
                print(line, file=newfile)

    return None

def update_parfiles(homepath, psr_info, use_newfile=False):
    """Apply parfile normalization across all pulsars.

    Args:
        homepath: Dataset root directory.
        psr_info: Pulsar info dict.
        use_newfile: If True, replace originals with .new outputs.

    Returns:
        None.
    """
    for psr in psr_info.keys():
        parfile = psr+".par"
        try:
            insert_missing_jumps(psr, parfilepath=os.path.join(homepath, psr, parfile), psr_info=psr_info)
            if use_newfile:
                os.system("\cp -f {}/{}/{} {}/{}/{}".format(homepath, psr, parfile, homepath, psr, parfile.replace(".par", ".orig")))
                os.system("mv {}/{}/{} {}/{}/{}".format(homepath, psr, parfile.replace(".par", ".new"), homepath, psr, parfile))
        except Exception as e:
            print(e)
            pass


    return None

def edit_parfiles(homepath, psr_info, use_newfile=False):
    """Normalize parfiles without exception handling.

    Args:
        homepath: Dataset root directory.
        psr_info: Pulsar info dict.
        use_newfile: If True, replace originals with .new outputs.

    Returns:
        None.
    """
    for psr in psr_info.keys():
        parfile = psr+".par"
        insert_missing_jumps(psr, parfilepath=os.path.join(homepath, psr, parfile), psr_info=psr_info)
        if use_newfile:
            os.system("\cp -f {}/{}/{} {}/{}/{}".format(homepath, psr, parfile, homepath, psr, parfile.replace(".par", ".orig")))

            os.system("mv {}/{}/{} {}/{}/{}".format(homepath, psr, parfile.replace(".par", ".new"), homepath, psr, parfile))

    return None

def equatorial_to_ecliptic_par(parfile):
    """Extract ecliptic coordinates from an equatorial parfile.

    Args:
        parfile: Path to the .par file.

    Returns:
        Tuple ``(elong, elat, pmelong, pmelat, lines, drop_lines)`` or None on failure.
    """
    with open(parfile, 'r') as orig:
        lines = orig.readlines()
    try:
        raj = [line.rstrip("\n").split() for line in lines if "RAJ" in line]
        decj = [line.rstrip("\n").split() for line in lines if "DECJ" in line]
        pmra = [line.rstrip("\n").split() for line in lines if "PMRA" in line]
        pmdec = [line.rstrip("\n").split() for line in lines if "PMDEC" in line]
        posepoch = [line.rstrip("\n").split() for line in lines if "POSEPOCH" in line]
        radec = SkyCoord(ra=raj[0][1], dec=decj[0][1], unit=(units.hourangle,units.deg), pm_ra_cosdec=float(pmra[0][1])*units.mas/units.yr, pm_dec=float(pmdec[0][1])*units.mas/units.yr, equinox='J2000', obstime=Time(float(posepoch[0][1]), format='mjd'))
        elong = radec.barycentrictrueecliptic.lon.to_string(decimal=True)
        elat = radec.barycentrictrueecliptic.lat.to_string(decimal=True)
        pmelong = radec.barycentrictrueecliptic.pm_lon_coslat.value
        pmelat = radec.barycentrictrueecliptic.pm_lat.value
        eqterms = ["RAJ", "DECJ", "PMRA", "PMDEC"]
        drop_lines = [rem for rem, line in enumerate(lines) for eqterm in eqterms if eqterm in line ]
        return elong, elat, pmelong, pmelat, lines, drop_lines
    except Exception as e:
        print("Warning: {} for {}.".format(e, parfile))
        return None

def create_ecliptic_par(parfile, use_newfile):
    """Create an equatorial parfile from ecliptic coordinates.

    Args:
        parfile: Path to the .par file.
        use_newfile: If True, replace the original with the converted file.

    Returns:
        None.
    """
    try:
        raj, decj, pmra, pmdec, lines, drop_lines = ecliptic_to_equatorial_par(parfile)
        with open(parfile.replace(".par", "_eq.par"), 'w') as eqpar:
            for line in lines:
                line = line.rstrip("\n")
                if "PMELONG" in line:
                    line = "{:<15}{:<15} 1 0.00000".format("PMRA", pmra)
                if "PMELAT" in line:
                    line = "{:<15}{:<15} 1 0.00000".format("PMDEC", pmdec)
                if "ELONG" in line and not "PM" in line:
                    line = "{:<15}{:<15} 1 0.00000".format("RAJ", raj)
                if "ELAT" in line and not "PM" in line:
                    line = "{:<15}{:<15} 1 0.00000".format("DECJ", decj)
                print(line, file=eqpar)

        if use_newfile:
            os.system("\cp -f {} {}".format(parfile, parfile.replace(".par", "_dr1eqecl.par")))
            os.system("mv {} {}".format(parfile.replace(".par", "_eq.par"), parfile))

    except Exception as e:
        print("Warning: {}".format(e))

    return None

def ecliptic_to_equatorial_par(parfile):
    """Convert ecliptic coordinates in a parfile to equatorial values.

    Args:
        parfile: Path to the .par file.

    Returns:
        Tuple ``(raj, decj, pmra, pmdec, lines, drop_lines)`` or None on failure.
    """
    with open(parfile, 'r') as orig:
        lines = orig.readlines()
    try:
        elong = [line.rstrip("\n").split() for line in lines if line.startswith("ELONG")]
        elat = [line.rstrip("\n").split() for line in lines if "ELAT" in line]
        pmelong = [line.rstrip("\n").split() for line in lines if "PMELONG" in line]
        pmelat = [line.rstrip("\n").split() for line in lines if "PMELAT" in line]
        posepoch = [line.rstrip("\n").split() for line in lines if "POSEPOCH" in line]
        elongelat = SkyCoord(lon=elong[0][1], lat=elat[0][1], unit=(units.hourangle,units.deg),
                             pm_lon_coslat=float(pmelong[0][1])*units.mas/units.yr,
                             pm_lat=float(pmelat[0][1])*units.mas/units.yr,
                             frame=BarycentricTrueEcliptic,
                             equinox='J2000', obstime=Time(float(posepoch[0][1]), format='mjd'))
        raj = elongelat.icrs.ra.to_string(unit='hourangle', sep=":")
        decj = elongelat.icrs.dec.to_string(unit=units.deg, sep=":")
        pmra = elongelat.icrs.pm_ra_cosdec.value
        pmdec = elongelat.icrs.pm_dec.value
        eqterms = ["ELONG", "ELAT", "PMELONG", "PMELAT"]
        drop_lines = [rem for rem, line in enumerate(lines) for eqterm in eqterms if eqterm in line ]
        return raj, decj, pmra, pmdec, lines, drop_lines
    except Exception as e:
        print("Warning: {} for {}.".format(e, parfile))
        return None

def check_all_binary(parfile_lines):
    """Return True if a parfile contains a full binary parameter set.

    Args:
        parfile_lines: Parfile contents as a string.

    Returns:
        True if required binary parameters are present.
    """
    if 'PB' in parfile_lines and 'A1' in parfile_lines and 'PBDOT' in parfile_lines and ('XDOT' in parfile_lines or 'A1DOT' in parfile_lines):
        return True
    else:
        return False

def check_drop_pkpars(params, parfile_lines, pkpars=['H3', 'STIG']):
    """Remove PK parameters from a list when already present in a parfile.

    Args:
        params: List of parameter names to modify.
        parfile_lines: Parfile contents as a string.
        pkpars: List of PK parameters to drop when present.

    Returns:
        Updated parameter list.
    """
    for pkpar in pkpars:
        try:
            if pkpar in parfile_lines:
                params.remove(pkpar)
        except Exception as e:
            print("Warning: {}".format(e))

    return params

def add_params(psr, parfilepath, psr_info, params, start_line):
    """Insert missing parameters into a parfile at a given line.

    Args:
        psr: Pulsar name (unused).
        parfilepath: Path to the .par file.
        psr_info: Pulsar info dict (unused).
        params: List of parameter names to add.
        start_line: Line number at which to insert parameters.

    Returns:
        List of parameters actually written.
    """
    with open(parfilepath, 'r') as workingfile:
        parfile_lines = workingfile.read()
        pkpars=['H3', 'STIG']
        if 'EPS1' in parfile_lines and 'ECCDOT' in params:
            params.remove('ECCDOT')
            params.extend(['EPS1DOT', 'EPS2DOT'])

        if ('{}M2'.format(os.linesep) in parfile_lines or 'SINI' in parfile_lines) and 'H3' in params:
            params.remove('H3')
            params.remove('STIG')
            pkpars=['M2', 'SINI']
            params.extend(pkpars)

        if check_all_binary(parfile_lines):
            check_drop_pkpars(params, parfile_lines, pkpars=pkpars)

        for param in params:
            if param in parfile_lines:
                params.remove(param)

        bin_params = ['PB', 'A1', 'X', 'ECC', 'EPS', 'OM']
        for bin_par in bin_params:
            pars2remove = [s for s in params if s.startswith(bin_par)]
            if bin_par not in parfile_lines and any(pars2remove):
                for s in pars2remove:
                    params.remove(s)

    with open(parfilepath, 'r') as workingfile:
        with open(parfilepath.replace(".par","_addpar.par"), 'w') as newfile:
            line_no = 0
            for line in workingfile:
                if line_no == start_line:
                    for newparam in params:
                        newfile.write("{:<15}0    1    0{}".format(newparam, os.linesep))
                newfile.write(line)
                line_no += 1

    return params

def add_params_to_parfile(homepath, psr_info, newpars, use_newfile=False, branch_message="Auto commit"):
    """Add parameter templates to parfiles for all pulsars.

    Args:
        homepath: Dataset root directory.
        psr_info: Pulsar info dict.
        newpars: Iterable of parameter names to add.
        use_newfile: If True, replace originals with modified files.
        branch_message: Commit message prefix for git commits.

    Returns:
        None.
    """
    for psr in psr_info.keys():
        parfile = psr+".par"
        start_line = findlinesstartingwith(searchstring="TZRFRQ", filename=os.path.join(homepath, psr, parfile))[0]
        added_params = add_params(psr, parfilepath=os.path.join(homepath, psr, parfile), psr_info=psr_info, params=list(newpars), start_line=start_line)

        if use_newfile:
            os.system("\cp -f {}/{}/{} {}/{}/{}".format(homepath, psr, parfile, homepath, psr, parfile.replace(".par", ".orig")))
            os.system("mv {}/{}/{} {}/{}/{}".format(homepath, psr, parfile.replace(".par", "_addpar.par"), homepath, psr, parfile))
            git.add(update=True)
            git.commit(message="{} Added {} for {}".format(branch_message, " and ".join(", ".join(added_params).rsplit(", ", 1)), psr))
            print("{} Added {} for {}".format(branch_message, " and ".join(", ".join(added_params).rsplit(", ", 1)), psr))

    return None

def run_full_notebook_fixes(psr_info: dict, homepath: str, magic_add_sysflag: bool = True, time_window: float = 0.0) -> None:
    """Convenience wrapper mirroring the notebook flow (tim flags + overlap removal + par updates)."""
    if magic_add_sysflag:
        update_timfiles(psr_info, homepath, magic_add_sysflag=True)
    # Overlap removal if you provide overlapped_timfiles and a time_window > 0
    if time_window and time_window > 0:
        remove_overlap(psr_info, homepath, overlapped_timfiles, time_window=time_window)
