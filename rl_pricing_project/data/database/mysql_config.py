import mysql.connector
from mysql.connector import Error
import os
from typing import Optional, Dict, Any

class MySQLConfig:
    """Configuration et gestion de la connexion MySQL pour RL_DATA_BASE"""
    
    def __init__(self):
        self.host = os.getenv('MYSQL_HOST', 'localhost')
        self.database = os.getenv('MYSQL_DATABASE', 'rl_data_base')
        self.user = os.getenv('MYSQL_USER', 'root')
        self.password = os.getenv('MYSQL_PASSWORD', '')
        self.port = os.getenv('MYSQL_PORT', '3306')
        
        self.connection = None
        self._test_connection()
    
    def _test_connection(self):
        """Teste la connexion à la base de données"""
        try:
            conn = self.get_connection()
            if conn.is_connected():
                print(f"✅ Connecté à MySQL: {self.database}")
                conn.close()
        except Error as e:
            print(f"❌ Erreur connexion MySQL: {e}")
    
    def get_connection(self):
        """Établit une connexion à la base de données"""
        try:
            connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            return connection
        except Error as e:
            print(f"Erreur connexion MySQL: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Exécute une requête SQL avec gestion d'erreurs"""
        connection = self.get_connection()
        cursor = None
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
            else:
                connection.commit()
                result = cursor.lastrowid
            
            return result
            
        except Error as e:
            print(f"Erreur requête SQL: {e}")
            connection.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

# Singleton pour la configuration DB
db_config = MySQLConfig()