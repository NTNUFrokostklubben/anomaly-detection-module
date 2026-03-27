CREATE TABLE IF NOT EXISTS images (
    img_id          TEXT PRIMARY KEY NOT NULL UNIQUE,  --full name of the image e.g. HX-14365_073_001_14822.tif
    prefix          TEXT NOT NULL,              --Just the prefix e.g. HX-14365
    project         TEXT,              --Project name from SKAVL
    line            INTEGER NOT NULL,           -- Line number from file e.g. 073
    line_number     INTEGER NOT NULL,           -- number of the image in relation to the line e.g. 001
    abs_number      INTEGER                     -- absolute number of the image in relation to the project e.g. 14822
);

CREATE TABLE IF NOT EXISTS artifact_datapoints(
    img_id      TEXT PRIMARY KEY NOT NULL,          --foreign key from image table, uniquely identifying
    dtype       TEXT,                               -- e.g. 'float32'
    shape       TEXT NOT NULL,                      -- e.g. '(45000 ,x_start, y_start)' x_end, y_end = x,y + offset.
    offset      INTEGER NOT NULL,                   -- the offset from x and y, describes the size of the block.
    data        BLOB NOT NULL,                      -- Data from artifact detecting algorithm.
    FOREIGN KEY (img_id) references images(img_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifact_candidates(
    coord_x     INTEGER NOT NULL CHECK (coord_x >= 0),  --  x coordinates for the artifact block
    coord_y     INTEGER NOT NULL CHECK (coord_y >= 0),  --  y coordinates for the artifact block
    img_id      TEXT NOT NULL,
    color_value REAL NOT NULL,                          -- the color value of the block
    diff_value  REAL NOT NULL,                          -- the diff value of the block
    offset      INTEGER NOT NULL,                       -- offset from x and y coords, describes the size of the block.

    FOREIGN KEY (img_id) REFERENCES artifact_datapoints(img_id) ON DELETE CASCADE,
    PRIMARY KEY  (coord_y, coord_x, img_id)
);

CREATE TABLE IF NOT EXISTS analysis_data(
    img_id          TEXT NOT NULL,  --full name of the image e.g. HX-14365_073_001_14822.tif
    analysis_type   TEXT CHECK( analysis_type in('color_avg', 'water_mask', 'artifact')) NOT NULL, --type of analysis
    confidence      REAL NOT NULL,       -- the level of confidence of the analysis
    FOREIGN KEY (img_id) references images(img_id) ON DELETE CASCADE,
    PRIMARY KEY (img_id, analysis_type)
)