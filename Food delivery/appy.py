from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum as _sum, desc
from flask import Flask, render_template
import os

app = Flask(__name__)

def get_analysis_data():
    """Perform analysis and return data as dictionaries"""
    # 1. Start Spark
    spark = SparkSession.builder \
        .appName("Online Food Delivery Analysis") \
        .getOrCreate()

    # 2. Load data
    df = spark.read.csv("food_orders.csv", header=True, inferSchema=True)
    
    # Convert raw data to list of dictionaries
    raw_data = [row.asDict() for row in df.collect()]
    
    # 3. Total orders per city
    orders_per_city = df.groupBy("city").agg(count("*").alias("total_orders"))
    orders_per_city_data = [row.asDict() for row in orders_per_city.collect()]
    
    # 4. Average order amount per city
    avg_amount_city = df.groupBy("city").agg(
        avg("order_amount").alias("avg_order_amount")
    )
    avg_amount_data = [row.asDict() for row in avg_amount_city.collect()]
    
    # 5. Average delivery time per city
    avg_delivery_city = df.groupBy("city").agg(
        avg("delivery_time_mins").alias("avg_delivery_time")
    )
    avg_delivery_data = [row.asDict() for row in avg_delivery_city.collect()]
    
    # 6. Most popular cuisine overall (by order count)
    cuisine_popularity = df.groupBy("cuisine").agg(
        count("*").alias("order_count")
    ).orderBy(desc("order_count"))
    cuisine_data = [row.asDict() for row in cuisine_popularity.collect()]
    
    # 7. Top 3 restaurants by total revenue
    top_restaurants = df.groupBy("restaurant").agg(
        _sum("order_amount").alias("total_revenue")
    ).orderBy(desc("total_revenue")).limit(3)
    top_restaurants_data = [row.asDict() for row in top_restaurants.collect()]
    
    # 8. High-value customers (total spend > 1000)
    customer_spend = df.groupBy("user_id").agg(
        _sum("order_amount").alias("total_spent")
    ).orderBy(desc("total_spent"))
    
    high_value_customers = customer_spend.filter(col("total_spent") > 1000)
    high_value_data = [row.asDict() for row in high_value_customers.collect()]
    
    spark.stop()
    
    return {
        'raw_data': raw_data,
        'orders_per_city': orders_per_city_data,
        'avg_amount_city': avg_amount_data,
        'avg_delivery_city': avg_delivery_data,
        'cuisine_popularity': cuisine_data,
        'top_restaurants': top_restaurants_data,
        'high_value_customers': high_value_data
    }

@app.route('/')
def index():
    """Main page displaying all analysis results"""
    data = get_analysis_data()
    return render_template('index.html', **data)

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
