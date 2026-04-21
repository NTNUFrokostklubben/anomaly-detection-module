import logging
import sqlite3 as sql
from os.path import exists
import atexit
from pathlib import Path

from entity.anomaly.ProjectMetadata import ProjectMetadata
from utils.string_manip import slice_image_name
import numpy as np
from entity.enums.analysis_t import AnalysisType

import sys
logger = logging.getLogger("database.connector")

class DbConnector:
    """
        Represents a singleton database connection, holds methods for CRUD operations.
        Attributes:
            _instance (DBConnector): the singleton DBConnector instance
            _conn (sql.Connection): the database connection.

    """
    _instance = None  # Class variable to store the single instance
    _conn = None        # holds the database connection.

    @staticmethod
    def _get_schema_path():
        """
        Static method for getting the schema path based on if project is run as dev or exe

        Returns:
            Schema path
        """
        if getattr(sys, 'frozen', False):
            return Path(sys._MEIPASS) / 'utils' / 'schema.sql'
        return Path(__file__).parent / 'schema.sql'

    @staticmethod
    def _get_db_path():
        """
        Static method for getting the db path based on if project is ran as dev or exe

        Returns:
            path to db
        """
        if getattr(sys, 'frozen', False):
            return Path(sys._MEIPASS) / 'database.db'
        return Path(__file__).parent.parent.parent / 'database.db'



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
            _sql_file = self._get_schema_path()    #Finds path to this files parent, then navigates to schema
            _db_file = self._get_db_path()
            if not exists(_sql_file):
                raise FileNotFoundError('sql schema not found')
            try:
                sql_script = ""
                with open(_sql_file, 'r') as f:
                    sql_script = f.read()

                self._conn = sql.connect(_db_file, check_same_thread=False)
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
            prefix, line, line_number, abs_number = slice_image_name(img_file_name)
            cursor.execute(
                """INSERT INTO images(img_id, prefix, line, line_number, abs_number)
                   VALUES (?, ?, ?, ?, ?)""",
                (img_file_name, prefix, line, line_number, abs_number))
            self.commit()
            logger.info("Added image %s", img_file_name,
                        extra={"operation": "INSERT", "table": "images"})
            return True
        except sql.DatabaseError as e:
            logger.error("Error: application failed to write to database with error: %s", e,
                         extra={"operation": "INSERT", "table": "images"})
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
            img_data = cursor.execute("""SELECT *
                                         FROM images
                                         WHERE img_id = ? """, (img_file_name,)).fetchone()
            if img_data is None:
                self.add_image(img_file_name)
        except sql.DatabaseError as e:
            logger.error(
                "Error: application failed to read from database with error: %s", e,
                extra={"operation": "SELECT", "table": "images"}
            )
            return False
        try:
            cursor.execute(
                    """
                           REPLACE INTO artifact_datapoints(img_id, dtype, shape, offset, data) VALUES(?, ?, ?, ?, ?)
                           """,(img_file_name, str(data.dtype), str(data.shape), offset, blob))

            self.commit()
            return True
        except sql.DatabaseError as e:
            logger.error(
                "Error: application failed to write to database with error: %s", e,
                extra={"operation": "REPLACE INTO", "table": "artifact_datapoints"}
                         )
            return False

    def add_artifact_candidate(self, img_file_name: str, color: float, diff: float, offset: int,
                               coords: tuple[int, int]) -> bool:
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
            img_data = cursor.execute("""SELECT *
                                         FROM images
                                         WHERE img_id = ? """, (img_file_name,)).fetchone()
            if img_data is None:
                self.add_image(img_file_name)
        except sql.DatabaseError as e:
            logger.error(
            "Error: application failed to read from database with error: %s", e,
            extra={"operation": "SELECT", "table": "images"}
            )
            return False
        try:

            cursor.execute(
                """ REPLACE INTO artifact_candidates(coord_x, coord_y, img_id, color_value, diff_value, offset)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (coords[0], coords[1], img_file_name, color, diff, offset))
            self.commit()
            return True
        except sql.DatabaseError as e:
            logger.error(
                "Error: application failed to write to database with error: %s", e,
                extra={"operation": "REPLACE INTO", "table": "artifact_candidates"}
            )
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
                           DELETE
                           FROM artifact_datapoints
                           WHERE img_id IN (SELECT img_id FROM images WHERE line = ? AND prefix = ?)

                           """, (line, prefix,))
            self.commit()
            return True
        except sql.DatabaseError as e:
            logger.error("Error: application failed to remove line %s from project %s in database with error: %s", line, prefix ,e ,
                extra = {"operation": "DELETE", "table": "artifact_candidates"}
                         )
            return False

    def add_project(self, project_metadata: ProjectMetadata):
        """

        Args:
            project_metadata: Project Metadata Entity

        Returns:

        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                           REPLACE INTO projects (project_name, sosi_path, image_folder_path, sosi_water_path)
                           VALUES (?, ?, ?, ?)
                           """, (project_metadata.project_name, project_metadata.sosi_path,
                                 project_metadata.image_folder_path, project_metadata.sosi_water_mask_path))
            self.commit()
            return True
        except Exception as e:
            logger.error("Error: application failed to write to database with error: %s", e,
                         extra={"operation": "REPLACE INTO", "table": "projects"}
                         )
            return False

    def increment_project_image_intex(self, project_name: str):
        """
        Updates the last processed image index for a given project name

        Args:
            project_name: Project name to increment last processed image index

        Returns:
            True or False
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                           UPDATE projects
                           SET last_processed_image_index = last_processed_image_index + 1
                           WHERE project_name = ?
                           """, (project_name,))
            self.commit()
            return True
        except Exception as e:
            logger.error("Error: application failed to write to database with error: %s", e,
                         extra={"operation": "UPDATE", "table": "projects"}
                         )
            return False

    def get_project(self, project_name: str) -> ProjectMetadata | None:
        """
        Gets project data stored in DB

        Args:
            project_name: project name string

        Returns:
            tuple: (project_name, sosi_path, image_folder_path)

        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                           SELECT *
                           FROM projects
                           WHERE project_name = ?
                           """, (project_name,))
            result = cursor.fetchone()
            if result is None:
                return None
            return ProjectMetadata.from_row(cursor.fetchone())
        except Exception as e:
            logger.error("Error: application failed to read from database with error: %s", e,
                         extra={"operation": "SELECT", "table": "projects"}
                         )
            return None


    def get_artifact_data_line(self, prefix: str, line: int, shape: str) -> list[np.ndarray] | None:

        """
        Get the computed artifact data for a line from the database.
        :param shape: the shape of the data, ensures data of different resolution don't get compared.
        :param prefix: the prefix of the artifact data.
        :param line:   the line of the artifact data to get.
        :return: the computed artifact data for the line from the database or None
        """
        try:
            cursor = self._conn.cursor()
            rows = cursor.execute("""
                                  SELECT dtype, shape, data
                                  FROM artifact_datapoints
                                  WHERE img_id IN (SELECT img_id
                                                   FROM images
                                                   WHERE line = ?
                                                     AND prefix = ?)
                                      AND shape = ?
                                  """, (line, prefix, shape, ))
            self.commit()
            return [np.frombuffer(r[2], dtype=r[0]).reshape(eval(r[1])) for r in rows]
        except sql.DatabaseError as e:
            logger.error("Error: application failed to read from database with error: %s", e,
                         extra={"operation": "SELECT", "table": "artifact_datapoints"}
                         )
            return None

    def add_analysis(self, img_file_name: str, analysis_type: AnalysisType, confidence_lvl: float  ):
        """
        Add an analysis to the database.
        :param img_file_name:  the file name of the image.
        :param analysis_type:  the type of analysis to add.
        :param confidence_lvl: the confidence level of the analysis to add.
        :return: True for success, False for failure.
        """
        try:
            cursor = self._conn.cursor()
            img_data = cursor.execute("""SELECT *
                                         FROM images
                                         WHERE img_id = ? """, (img_file_name,)).fetchone()

            if img_data is None:
                self.add_image(img_file_name)

            rows = cursor.execute(
                """ 
            REPLACE INTO analysis_data(img_id, analysis_type, confidence) values(?,?,?)  
                                                        -- Replace into is shorthand for insert or replace.
                           """, (img_file_name, analysis_type.value, confidence_lvl))
            self.commit()
            return True
        except sql.DatabaseError as e :
            logger.error("Error: application failed to write to database with error: %s", e,
                         extra={"operation": "REPLACE INTO", "table": "analysis_data"}
                         )
            return False

    def get_all_img_over_confidence(self, project:str, confidence: float ) -> list[tuple[str, float]] | None:
        """
        Fetch all image file names over a confidence threshold based on project.
        :param project:
        :param confidence:
        :return:
        """
        try:
            cursor = self._conn.cursor()
            confidence_data = cursor.execute("""
                                             SELECT a.img_id, MAX(a.confidence) as max_confidence
                                             FROM analysis_data a
                                             JOIN images i ON a.img_id = i.img_id
                                             WHERE i.project = ?
                                             GROUP BY a.img_id
                                             HAVING MAX(a.confidence) >= ?
                                             """, (project, confidence)).fetchall()
            self.commit()
            return confidence_data
        except sql.DatabaseError as e:
            logger.error("Error: application failed to read from database with error: %s", e,
                         extra={"operation": "SELECT", "table": "analysis_data"}
                         )
            return None

    def get_all_analysis_img(self, img_file_name: str) -> list[tuple[AnalysisType, float]] | None:
        """
        Fetch all analysis data for an image
        :param img_file_name:  the file name of the image.
        :return: the list of analysis data for an image.
        """
        try:
            cursor = self._conn.cursor()
            analysis_data = cursor.execute("""SELECT analysis_type, confidence FROM analysis_data 
                                                    WHERE img_id = ?
                                             """, (img_file_name, )).fetchall()
            self.commit()
            return analysis_data
        except sql.DatabaseError as e:
            logger.error("Error: application failed to read from database with error: %s", e,
                         extra={"operation": "SELECT", "table": "analysis_data"}
                         )
            return None


    def get_max_confidence_img(self, img_file_name: str) -> float | None:
        """
        Get the largest analysis confidence for an image
        :param img_file_name:  the file name of the image.
        :return: the largest analysis confidence for an image.
        """
        try:
            cursor = self._conn.cursor()
            analysis_data = cursor.execute(""" 
                                           SELECT MAX(confidence) FROM analysis_data WHERE img_id = ?
                                             """, (img_file_name, )).fetchone()
            self.commit()
            return float(analysis_data[0])
        except sql.DatabaseError as e:
            logger.error("Error: application failed to read from database with error: %s", e,
                         extra={"operation": "SELECT", "table": "analysis_data"}
                         )
            return None


    def commit(self):
        """
        Shorthand method for commiting SQL query.
        :return:
        """
        try:
            self._conn.commit()
        except sql.DatabaseError as e:
            logger.error("Error: application failed to commit to database with error: %s", e,
                         extra={"operation": "COMMIT", "table": ""}
                         )

    def __del__(self):
        """
        Should close the connection on process stop.
        """
        atexit.register(self._conn.close)
