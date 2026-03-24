import sqlite3 as sql
from os.path import exists
import atexit
from pathlib import Path
from utils.string_manip import slice_image_name
import numpy as np


class DbConnector:
    """
        Represents a singleton database connection, holds methods for CRUD operations.
        Attributes:
            _instance (DBConnector): the singleton DBConnector instance
            _conn (sql.Connection): the database connection.

    """
    _instance = None  # Class variable to store the single instance
    _conn = None        # holds the database connection.
    _sql_file = Path(__file__).parent / 'schema.sql'     #Finds path to this files parent, then navigates to schema
    _db_file = 'database.db'


    def __new__(cls, *args, **kwargs):
        """
        Create new instance of singleton if not exists, otherwise, serve existing singleton
        :param args:
        :param kwargs:
        """
        if cls._instance is None:
            # Create a new instance if one doesn't exist
            cls._instance = super(DbConnector, cls).__new__(cls, *args, **kwargs)
        cls.init(cls._instance)
        return cls._instance

    def init(self):
        """
        Create the database connection
        """
        if self._conn is None:
            if not exists(self._sql_file):
                raise FileNotFoundError('sql schema not found')
            try:
                sql_script = ""
                with open(self._sql_file, 'r') as f:
                        sql_script = f.read()

                self._conn = sql.connect(self._db_file)
                self._conn.execute("PRAGMA foreign_keys = ON")
                self._conn.execute("PRAGMA journal_mode=WAL")
                cursor = self._conn.cursor()
                cursor.executescript(sql_script)
                self.commit()

            except sql.DatabaseError as e:
                print(e)


    def add_image(self, img_file_name) -> bool:
        """
        Add an image to the database. Contains no image data, just metadata.
        :param img_file_name: the file name of the image.
        :return: True for success, False for failure.
        """
        try:
            cursor = self._conn.cursor()
            prefix, line, line_number, abs_number =  slice_image_name(img_file_name)
            cursor.execute(
                """INSERT INTO images(img_id, prefix, line, line_number, abs_number) VALUES(?, ?, ?, ?, ?)""",
                (img_file_name, prefix, line, line_number, abs_number))
            self.commit()
            return True
        except sql.DatabaseError as e:
            return False


    def add_artifact_data(self, img_file_name: str, data: np.ndarray, offset: int):
        """
        Add the data from an artifact detection run on an image to the database.
        :param img_file_name: the file name of the image.
        :param data: the array of data to add to the db, should be orders of magnitude smaller than total pixels in image.
        :param offset: the amount of offset used when calculating artifacts.
        :return:
        """
        try:
            cursor = self._conn.cursor()
            blob = data.tobytes()
            img_data = cursor.execute("""SELECT * FROM images WHERE img_id = ? """, (img_file_name,)).fetchone()
            if img_data is None:
                self.add_image(img_file_name)

            cursor.execute("""
                           INSERT OR REPLACE INTO artifact_datapoints(img_id, dtype, shape, offset, data) VALUES(?, ?, ?, ?, ?)
                           """,(img_file_name, str(data.dtype), str(data.shape), offset, blob))

            self.commit()
            return True
        except sql.DatabaseError as e:
            print(e)
            return False

    def add_artifact_candidate(self, img_file_name: str, color: float, diff: float, offset: int, coords: tuple[int, int]) -> bool:
        """
        Add an artifact candidate to the database.
        :param img_file_name: The file name of the image.
        :param color: the color value of the artifact candidate.
        :param diff:  the biggest diff between this candidate in this image and the same candidate in some other image.
        :param offset: the amount of offset used when calculating artifacts.
        :param coords: the x,y coordinates of the upper left corner of the artifact candidate.
        :return: true for success, false for failure.
        """
        try:
            cursor = self._conn.cursor()
            img_data = cursor.execute("""SELECT * FROM images WHERE img_id = ? """, (img_file_name,)).fetchone()
            if img_data is None:
                self.add_image(img_file_name)

            cursor.execute(""" INSERT INTO artifact_candidates(coord_x, coord_y, img_id, color_value, diff_value, offset) 
                           VALUES(?, ?, ?, ?, ?, ?)""",
                           (coords[0],coords[1],img_file_name,color,diff,offset))
            self.commit()
            return True
        except sql.DatabaseError as e:
            print(e)
            return False

    def delete_artifact_data_line(self, prefix: str, line: int) -> bool:
        """
        Delete the artifact data line from the database.
        :param prefix: the prefix of the artifact data.
        :param line:  the line of the artifact data to delete.
        :return:  True for success, False for failure.
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(""" 
                            DELETE FROM artifact_datapoints                                                                                           
                            WHERE img_id IN (SELECT img_id FROM images WHERE line = ? AND prefix = ?)                                                                 
                                                                     
                           """, (line, prefix,))
            self.commit()
            return True
        except sql.DatabaseError:
            return False


    def get_artifact_data_line(self, prefix: str, line: int)-> list[np.ndarray] | None:

        """
        Get the computed artifact data for a line from the database.
        :param prefix: the prefix of the artifact data.
        :param line:   the line of the artifact data to get.
        :return: the computed artifact data for the line from the database.
        """
        try:
            cursor = self._conn.cursor()
            rows = cursor.execute("""
                            SELECT dtype, shape, data
                            FROM artifact_datapoints
                            WHERE img_id IN (
                                SELECT img_id FROM images
                                WHERE line = ? AND prefix = ?)
                           """, (line, prefix,))
            self.commit()
            return [np.frombuffer(r[2], dtype=r[0]).reshape(eval(r[1])) for r in rows]
        except sql.DatabaseError:
            return None


    def commit(self):
        """
        Shorthand method for commiting SQL query.
        :return:
        """
        self._conn.commit()

    def __del__(self):
        """
        Should close the connection on process stop.
        """
        atexit.register(self._conn.close)
